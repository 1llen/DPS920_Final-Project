import cv2
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

from preprocessing import load_image, preProcessing


def zoom(img):
    aug = iaa.Affine(scale=(1.0, 1.3))
    return aug.augment_image(img)


def pan(img):
    aug = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
    return aug.augment_image(img)


def brightness(img):
    aug = iaa.Multiply((0.4, 1.2))
    return aug.augment_image(img)


def rotate(img, degrees=8):
    aug = iaa.Affine(rotate=(-degrees, degrees))
    return aug.augment_image(img)


def flip(img, steering):
    return cv2.flip(img, 1), -steering


def random_augment(image_path, steering, p=0.5):
    img = load_image(image_path)

    # Randomly apply augmentations
    if np.random.rand() < p:
        img = pan(img)
    if np.random.rand() < p:
        img = zoom(img)
    if np.random.rand() < p:
        img = brightness(img)
    if np.random.rand() < p:
        img = rotate(img)
    if np.random.rand() < p:
        img, steering = flip(img, steering)

    return img, steering


if __name__ == '__main__':
    import data_loader

    DATA_DIRS = ['data_forward', 'data_reverse']

    data = data_loader.load_all(DATA_DIRS)

    # Pick a sample
    turning = data[data['Steering'].abs() > 0.15]
    row = turning.sample(1, random_state=1).iloc[0]
    path, angle = row['ImagePath'], row['Steering']

    # 
    original = load_image(path)
    flipped, flipped_angle = flip(original, angle)

    # Augmentations
    named = [
        (original, f'original  steering={angle:.3f}'),
        (zoom(original), 'zoom'),
        (pan(original), 'pan'),
        (brightness(original), 'brightness'),
        (rotate(original), 'rotate'),
        (flipped, f'flip  steering={flipped_angle:.3f}'),
    ]

    # batch
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, (img, title) in zip(axes.flatten(), named):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    #  fully random augmentations
    fig, axes = plt.subplots(2, 4, figsize=(16, 5))
    for ax in axes.flatten():
        img, new_angle = random_augment(path, angle)
        ax.imshow(img)
        ax.set_title(f'{new_angle:.3f}', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # fix shape
    img, new_angle = random_augment(path, angle)
    out = preProcessing(img)
    print(f'[INFO] augmented -> preprocessed shape {out.shape}, '
          f'range {out.min():.3f} to {out.max():.3f}')
    assert out.shape == (66, 200, 3)