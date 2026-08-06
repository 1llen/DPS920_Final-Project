import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_image, preProcessing

def zoom(img):
    h, w = img.shape[:2]
    scale = np.random.uniform(1.0, 1.3)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(img, M, (w, h))


def pan(img):
    h, w = img.shape[:2]
    tx = np.random.uniform(-0.1, 0.1) * w
    ty = np.random.uniform(-0.1, 0.1) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))


def brightness(img):
    return cv2.convertScaleAbs(img, alpha=np.random.uniform(0.4, 1.2))


def rotate(img):
    h, w = img.shape[:2]
    angle = np.random.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def flip(img, steering):
    """Mirror the image horizontally and negate the steering angle to match.

    Args:
        img: The image to flip.
        steering: The steering angle associated with the image.

    Returns:
        A tuple of the horizontally flipped image and the negated steering
        angle (a left turn in the mirrored frame is a right turn in reality).
    """
    return cv2.flip(img, 1), -steering


def random_augment(img, steering, p=0.5, use_rotate=False):
    """Apply a random subset of augmentations to an image/steering pair.

    Each of pan, zoom, brightness, optional rotate, and flip is applied
    independently with probability `p`.

    Args:
        img: The image to augment.
        steering: The steering angle associated with the image.
        p: The probability of applying each individual augmentation.
        use_rotate: Whether the rotate augmentation is eligible to be applied.

    Returns:
        A tuple of the augmented image and the steering angle updated to
        match (only the flip augmentation changes it).
    """
    if np.random.rand() < p:
        img = pan(img)
    if np.random.rand() < p:
        img = zoom(img)
    if np.random.rand() < p:
        img = brightness(img)
    if use_rotate and np.random.rand() < p:
        img = rotate(img)
    if np.random.rand() < p:
        img, steering = flip(img, steering)

    return img, steering


if __name__ == '__main__':
    import data_loader

    DATA_DIRS = ['data_forward', 'data_reverse']

    data = data_loader.load_all(DATA_DIRS)

    # Pick a random image that has a non-trivial steering angle
    turning = data[data['Steering'].abs() > 0.15]
    row = turning.sample(1, random_state=1).iloc[0]
    angle = row['Steering']

    # Show the original and each augmentation applied to it, with titles
    original = load_image(row['ImagePath'])
    flipped, flipped_angle = flip(original, angle)

    named = [
        (original, f'original  steering={angle:.3f}'),
        (zoom(original), 'zoom'),
        (pan(original), 'pan'),
        (brightness(original), 'brightness'),
        (rotate(original), 'rotate'),
        (flipped, f'flip  steering={flipped_angle:.3f}'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, (img, title) in zip(axes.flatten(), named):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Show a grid of random augmentations of the same original image
    fig, axes = plt.subplots(2, 4, figsize=(16, 5))
    for ax in axes.flatten():
        # Randomly augment the original image, then preprocess it for the model
        img, new_angle = random_augment(original, angle)
        ax.imshow(img)
        ax.set_title(f'{new_angle:.3f}', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Show that the preprocessed output has the right shape and range
    img, new_angle = random_augment(original, angle)
    out = preProcessing(img)
    print(f'[INFO] augmented -> preprocessed shape {out.shape}, '
          f'range {out.min():.3f} to {out.max():.3f}')
    assert out.shape == (66, 200, 3)