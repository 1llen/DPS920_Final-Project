import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


COLUMNS = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']


def get_filename(path):
    return path.strip().replace('\\', '/').split('/')[-1]


def load_log(csv_path, image_dir=None):
    csv_path = os.path.abspath(csv_path)
    # If no image_dir is provided, assume images are in the same directory as the CSV file
    if image_dir is None:
        image_dir = os.path.join(os.path.dirname(csv_path), 'IMG')

    data = pd.read_csv(csv_path, names=COLUMNS)
    data['Center'] = data['Center'].apply(get_filename)
    data['ImagePath'] = data['Center'].apply(
        lambda name: os.path.join(image_dir, name))

    # Label by parent folder - every session's file is called driving_log.csv
    session = os.path.basename(os.path.dirname(csv_path))
    s = data['Steering']
    print(f'[INFO] {session}: {len(data)} rows  '
          f'left={(s < 0).mean() * 100:.1f}%  '
          f'right={(s > 0).mean() * 100:.1f}%  '
          f'sum={s.sum():.1f}')

    return data[['ImagePath', 'Steering']]


def load_all(data_dirs):
    frames = []
    for d in data_dirs:
        frames.append(load_log(os.path.join(d, 'driving_log.csv')))

    data = pd.concat(frames, ignore_index=True)
    print(f'[INFO] {len(data)} total samples from {len(data_dirs)} session(s)')
    return data


def check_images(data):
    # Check that all images exist
    missing = [p for p in data['ImagePath'] if not os.path.exists(p)]
    if missing:
        print(f'[WARN] {len(missing)} images missing! First: {missing[0]}')
    else:
        print('[INFO] all images found')
    return len(missing)


def describe(data):
    # Describe the data
    s = data['Steering']
    # Check steering distribution
    left, right, zero = (s < 0).sum(), (s > 0).sum(), (s == 0).sum()
    print(f'[INFO] left={left} ({left / len(s) * 100:.1f}%)  '
          f'right={right} ({right / len(s) * 100:.1f}%)  '
          f'zero={zero} ({zero / len(s) * 100:.1f}%)')
    print(f'[INFO] mean={s.mean():.4f}  min={s.min():.3f}  max={s.max():.3f}')


def draw_histogram(ax, data, bins, samples_per_bin=None, title=''):
    hist, _ = np.histogram(data['Steering'], bins=bins)
    centers = (bins[:-1] + bins[1:]) * 0.5
    width = (bins[1] - bins[0]) * 0.9

    ax.bar(centers, hist, width=width)
    if samples_per_bin is not None:
        ax.axhline(samples_per_bin, color='r', linestyle='--',
                   label=f'cap = {samples_per_bin}')
        ax.legend()

    ax.set_title(f'{title}\n{len(data)} samples, tallest bin = {hist.max()}')
    ax.set_xlabel('steering angle')
    ax.set_ylabel('frames')


def plot_comparison(before, after, n_bins=25, samples_per_bin=None):
    # Plot the before and after histograms
    _, bins = np.histogram(before['Steering'], n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    draw_histogram(axes[0], before, bins, samples_per_bin, 'before balancing')
    draw_histogram(axes[1], after, bins, samples_per_bin, 'after balancing')

    plt.tight_layout()
    plt.show()


def balance_data(data, n_bins=25, samples_per_bin=400, seed=42):
    # Balance the data
    rng = np.random.default_rng(seed)
    steering = data['Steering'].values
    _, bins = np.histogram(steering, n_bins)

    # Determine which data to remove
    remove_list = []
    for i in range(n_bins):
        in_bin = np.where((steering >= bins[i]) & (steering <= bins[i + 1]))[0]
        if len(in_bin) > samples_per_bin:
            rng.shuffle(in_bin)
            remove_list.extend(in_bin[samples_per_bin:])

    # Remove the rows
    balanced = data.drop(data.index[remove_list]).reset_index(drop=True)
    print(f'[INFO] removed {len(remove_list)}, kept {len(balanced)}')
    return balanced


if __name__ == '__main__':
    # Two data sets, forward and reverse track
    DATA_DIRS = ['data_forward', 'data_reverse']
    SAMPLES_PER_BIN = 400

    data = load_all(DATA_DIRS)
    check_images(data)
    describe(data)

    balanced = balance_data(data, samples_per_bin=SAMPLES_PER_BIN)
    describe(balanced)

    plot_comparison(data, balanced, samples_per_bin=SAMPLES_PER_BIN)