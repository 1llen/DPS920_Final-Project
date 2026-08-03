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

    layers.Conv2D(64, (3, 3)),
	layers.BatchNormalization(),
	layers.ELU(),
	
    layers.Conv2D(64, (3, 3)),
	layers.BatchNormalization(),
	layers.ELU(),

    layers.Flatten(),

    layers.Dense(100),
	layers.BatchNormalization(),
	layers.ELU(),

    layers.Dense(50),
	layers.BatchNormalization(),
	layers.ELU(),

    layers.Dense(10, activation='elu'),
    layers.Dense(1)
])

model.summary()

# TRAIN
opt = AdamW(learning_rate=0.001, weight_decay=0.004)
model.compile(optimizer=opt,
              loss='mse')


H = model.fit(train_generator,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_data=validation_data,
              epochs=EPOCHS,
              verbose=2)


# EVALUATE
save_model(model, 'model_alt.h5')

loss = model.evaluate(validation_data[0], validation_data[1])
baseline = np.mean(validation_data[1] ** 2)
print(f'validation MSE = {loss}')
print(f'always-predict-zero MSE = {baseline}')

plt.plot(np.arange(0, EPOCHS), H.history['loss'], label='loss')
plt.plot(np.arange(0, EPOCHS), H.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('loss_curve_alt.png')
plt.show()
