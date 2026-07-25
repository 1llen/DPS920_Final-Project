import tensorflow as tf, cv2, numpy as np, pandas as pd, matplotlib, PIL
print(tf.__version__, cv2.__version__, np.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))