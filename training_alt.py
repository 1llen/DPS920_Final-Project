import os
os.environ['TF_DISABLE_MKL'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['KMP_BLOCKTIME'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, regularizers
from tensorflow.keras.models import save_model
# Using legacy to speed up training on Mac
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf

import data_loader
from generator import split_data, load_images, batch_generator, load_validation_set

DATA_DIRS = ['data_forward', 'data_reverse']
SAMPLES_PER_BIN = 400
BATCH_SIZE = 100
STEPS_PER_EPOCH = 300
EPOCHS = 15


# DATA
data = data_loader.load_all(DATA_DIRS)
data_loader.check_images(data)
data = data_loader.balance_data(data, samples_per_bin=SAMPLES_PER_BIN)

X_train, X_test, y_train, y_test = split_data(data)

# Images
train_images = load_images(X_train)
train_generator = batch_generator(train_images, y_train, BATCH_SIZE, True)
validation_data = load_validation_set(X_test, y_test)


# MODEL
model = Sequential([
    layers.Conv2D(24, (3, 3), strides=(2, 2), input_shape=(66, 200, 3), activation="elu"),
    layers.SpatialDropout2D(rate=0.1),

    layers.Conv2D(36, (3, 3), strides=(2, 2)),
	layers.BatchNormalization(),
	layers.ELU(),
    layers.SpatialDropout2D(rate=0.1),

    layers.Conv2D(48, (3, 3), strides=(2, 2)),
	layers.BatchNormalization(),
	layers.ELU(),
    layers.SpatialDropout2D(rate=0.1),

    layers.Conv2D(96, (3, 3)),
	layers.BatchNormalization(),
	layers.ELU(),
	
    layers.Conv2D(128, (3, 3)),
	layers.BatchNormalization(),
	layers.ELU(),

    layers.Flatten(),

    layers.Dense(256),
	layers.BatchNormalization(),
	layers.ELU(),
    layers.Dropout(0.15),

	layers.Dense(256),
	layers.BatchNormalization(),
	layers.ELU(),
    layers.Dropout(0.15),

    layers.Dense(128),
	layers.BatchNormalization(),
	layers.ELU(),
    layers.Dropout(0.1),

	layers.Dense(64),
	layers.BatchNormalization(),
	layers.ELU(),
    layers.Dropout(0.05),

    layers.Dense(32, activation='elu'),
    layers.Dense(1)
])

model.summary()

# CUSTOM LOSS 

def weighted_mse(y_true, y_pred):
    """MSE loss that upweights samples with larger steering angles.

    Straight-driving frames (steering near zero) dominate the dataset, so a
    plain MSE would under-penalize errors on sharp turns. Each sample's
    squared error is scaled by 1 + 5 * |y_true|, so bigger true steering
    angles count more toward the loss.

    Args:
        y_true: Ground-truth steering angles.
        y_pred: Predicted steering angles.

    Returns:
        The scalar weighted mean squared error.
    """
    # Penalizing values close to zero
    weights = 1.0 + 5.0 * tf.abs(y_true)
    squared_error = tf.square(y_true - y_pred)
    return tf.reduce_mean(weights * squared_error)

# TRAIN
opt = AdamW(learning_rate=0.002, weight_decay=0.004)
model.compile(optimizer=opt,
			  loss=weighted_mse,
              metrics=['mse'])
callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_mse', factor=0.5, patience=1, min_lr=1e-6
)

H = model.fit(train_generator,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_data=validation_data,
              epochs=EPOCHS,
              callbacks=[callback],
              verbose=2)


# EVALUATE
save_model(model, 'model_alt.h5')

results = model.evaluate(validation_data[0], validation_data[1], return_dict=True)

validation_mse = results["mse"]
baseline_mse = np.mean(validation_data[1] ** 2)

print(f"validation MSE = {validation_mse}")
print(f"weighted validation loss = {results['loss']}")
print(f"always-predict-zero MSE = {baseline_mse}")

plt.plot(np.arange(0, EPOCHS), H.history["mse"], label="training MSE")
plt.plot(np.arange(0, EPOCHS), H.history["val_mse"], label="validation MSE")
plt.xlabel("Epoch")
plt.ylabel("Plain MSE")
plt.legend()
plt.savefig("loss_curve_alt.png")
plt.show()