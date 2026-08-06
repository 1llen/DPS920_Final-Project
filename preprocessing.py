import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'could not read image: {path}')
    # convert BGR to RGB for matplotlib
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preProcessing(img):
    """Crop, recolor, blur, resize, and normalize a raw RGB frame for the model.

    Args:
        img: An RGB image, as returned by load_image.

    Returns:
        A (66, 200, 3) YUV image with pixel values normalized to 0-1.
    """
    # convert to YUV
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # resize to 200x66
    img = cv2.resize(img, (200, 66))
    # normalize to 0-1
    img = img / 255

    return img


if __name__ == '__main__':
    import data_loader

    DATA_DIRS = ['data_forward', 'data_reverse']

    data = data_loader.load_all(DATA_DIRS)
    sample = data.sample(3, random_state=0)

    fig, axes = plt.subplots(3, 3, figsize=(12, 7))
    for row, (_, item) in enumerate(sample.iterrows()):
        original = load_image(item['ImagePath'])
        cropped = original[60:135, :, :]
        processed = preProcessing(original)

        axes[row][0].imshow(original)
        axes[row][0].set_title(f'original {original.shape}')

        axes[row][1].imshow(cropped)
        axes[row][1].set_title(f'cropped {cropped.shape}')

        axes[row][2].imshow(processed)
        axes[row][2].set_title(f'processed {processed.shape}')

        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

    out = preProcessing(load_image(data['ImagePath'].iloc[0]))
    print(f'[INFO] output shape {out.shape}, range {out.min():.3f} to {out.max():.3f}')
    assert out.shape == (66, 200, 3), 'shape must match the Nvidia model input'