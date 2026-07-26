import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from preprocessing import load_image, preProcessing
from augmentation import random_augment


def split_data(data, test_size=0.2, random_state=42):
    X = data['ImagePath'].values
    y = data['Steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    print(f'[INFO] train={len(X_train)}  validation={len(X_valid)}')
    return X_train, X_valid, y_train, y_valid


def load_images(image_paths):
    images = [load_image(p) for p in image_paths]
    megabytes = sum(img.nbytes for img in images) / 1e6
    print(f'[INFO] cached {len(images)} images in memory ({megabytes:.0f} MB)')
    return images


def batch_generator(images, steerings, batch_size, is_training):
    while True:
        batch_images = []
        batch_steerings = []

        for _ in range(batch_size):
            i = np.random.randint(0, len(images))

            if is_training:
                img, steering = random_augment(images[i], steerings[i])
            else:
                img, steering = images[i], steerings[i]

            batch_images.append(preProcessing(img))
            batch_steerings.append(steering)

        yield (np.asarray(batch_images, dtype=np.float32),
               np.asarray(batch_steerings, dtype=np.float32))


def load_validation_set(image_paths, steerings):
    images = [preProcessing(load_image(p)) for p in image_paths]
    print(f'[INFO] validation set built: {len(images)} images')

    return (np.asarray(images, dtype=np.float32),
            np.asarray(steerings, dtype=np.float32))


if __name__ == '__main__':
    import data_loader

    DATA_DIRS = ['data_forward', 'data_reverse']
    SAMPLES_PER_BIN = 400
    BATCH_SIZE = 100

    data = data_loader.load_all(DATA_DIRS)
    data = data_loader.balance_data(data, samples_per_bin=SAMPLES_PER_BIN)
    X_train, X_valid, y_train, y_valid = split_data(data)

    train_images = load_images(X_train)
    valid_images = load_images(X_valid)

    train_gen = batch_generator(train_images, y_train, BATCH_SIZE, True)
    valid_gen = batch_generator(valid_images, y_valid, BATCH_SIZE, False)

    x_batch, y_batch = next(train_gen)
    print(f'[INFO] train batch  {x_batch.shape}  {x_batch.dtype}  '
          f'range {x_batch.min():.3f} to {x_batch.max():.3f}')
    assert x_batch.shape == (BATCH_SIZE, 66, 200, 3)
    assert y_batch.shape == (BATCH_SIZE,)

    v_batch, _ = next(valid_gen)
    print(f'[INFO] valid batch  {v_batch.shape}  {v_batch.dtype}')

    # Estimate the time per batch
    start = time.time()
    for _ in range(10):
        next(train_gen)
    per_batch = (time.time() - start) / 10
    print(f'[INFO] {per_batch:.2f} s per batch of {BATCH_SIZE} '
          f'({per_batch / BATCH_SIZE * 1000:.1f} ms per image)')
    print(f'[INFO] estimated {per_batch * 100 / 60:.1f} min per epoch '
          f'at 100 steps (data loading only)')

    # Check the output of the generators
    fig, axes = plt.subplots(2, 5, figsize=(16, 5))
    for ax, img, angle in zip(axes[0], x_batch, y_batch):
        ax.imshow(img)
        ax.set_title(f'train {angle:.2f}', fontsize=9)
        ax.axis('off')
    for ax, img in zip(axes[1], v_batch):
        ax.imshow(img)
        ax.set_title('validation', fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Check the distribution of steering angles
    sample = np.concatenate([next(train_gen)[1] for _ in range(20)])
    print(f'[INFO] {len(sample)} augmented labels: mean={sample.mean():.4f}  '
          f'left={(sample < 0).mean() * 100:.1f}%  '
          f'right={(sample > 0).mean() * 100:.1f}%')