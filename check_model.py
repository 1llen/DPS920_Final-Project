import os
os.environ['TF_DISABLE_MKL'] = '1'

import numpy as np
from tensorflow.keras.models import load_model

import data_loader


print('=== MODEL WEIGHTS ===', flush=True)
model = load_model('model.h5')

bad = False
for i, w in enumerate(model.get_weights()):
    n_nan = int(np.isnan(w).sum())
    n_inf = int(np.isinf(w).sum())
    finite = w[np.isfinite(w)]
    peak = np.abs(finite).max() if finite.size else float('nan')
    flag = ''
    if n_nan or n_inf:
        flag = '  <-- BROKEN'
        bad = True
    elif peak > 100:
        flag = '  <-- suspiciously large'
        bad = True
    print(f'array {i:2d}  shape {str(w.shape):20s}  nan={n_nan:6d}  '
          f'inf={n_inf:6d}  max|w|={peak:.4g}{flag}', flush=True)

print(f'\nverdict: {"MODEL IS CORRUPT - retrain" if bad else "weights look sane"}\n',
      flush=True)


print('=== TRAINING LABELS ===', flush=True)
DATA_DIRS = ['data_forward', 'data_reverse']

data = data_loader.load_all(DATA_DIRS)
s = data['Steering'].values

print(f'nan labels: {int(np.isnan(s).sum())}', flush=True)
print(f'inf labels: {int(np.isinf(s).sum())}', flush=True)
print(f'range: {np.nanmin(s):.3f} to {np.nanmax(s):.3f}', flush=True)
print(f'outside -1..1: {int((np.abs(s) > 1).sum())}', flush=True)

missing = data_loader.check_images(data)