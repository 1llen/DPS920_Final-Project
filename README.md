# Environment Setup

For our implementation, we used Python 3.8.

We created an environment using conda and the provided package requirements file:

```
conda create --name venv_CV-Project --file .\package_list.txt

conda activate venv_CV-Project
```

A number of the requirements had to be installed using pip instead of conda, as they were indicated in the package_list.txt:

```
pip install pandas==1.2.4 scikit-learn==0.24.2 scikit-image==0.18.3 imgaug==0.4.0 joblib==1.0.1 opencv-python matplotlib pillow

pip install flask==1.1.2 werkzeug==2.0.1 jinja2==3.0.1 itsdangerous==2.0.1 markupsafe==2.0.1 python-socketio==4.2.1 python-engineio==3.8.2.post1 flask-socketio==3.3.1 eventlet==0.25.1
```

# Project Structure

## 4. Project structure

```
project/
  data_forward/            # forward-direction recording
    driving_log.csv
    IMG/
  data_reverse/            # reverse-direction recording
    driving_log.csv
    IMG/

  data_loader.py           # loading and balancing data
  preprocessing.py         # crop / YUV / blur / resize / normalise
  augmentation.py          # pan, zoom, brightness, flip, and rotate training data
  generator.py             # train/validation split and batch generator
  training.py              # model training loop

  TestSimulation.py        # provided testing code, left unmodified
  model.h5                 # trained model
```

The recording folders are excluded from version control since they are exceptionally large, and were shared by compressing and uploading them for developers to download.

Each module can be ran directly to verify its own stage:

| Command | Verifies |
|---|---|
| `python data_loader.py` | Steering histograms before and after balancing |
| `python preprocessing.py` | Crop bounds and output shape |
| `python augmentation.py` | Each transformation |
| `python generator.py` | Batch shapes, augmented vs clean batches, throughput |

# Data Collection

Data was collected using the provided simulator from Udacity. We manually drove around the track using mouse controls, and split the recording into 2 sessions: forward and reverse. This is because the track had a bias towards turns in one direction since it was a loop, so to balance that out, we needed to also drive the opposite way. The two sessions were recorded into two separate folders:

| Session | Frames | Left | Right |
|---|---|---|---|
| `data_forward` | 4,494 | 60.9% | 7.4% |
| `data_reverse` | 4,900 | 12.6% | 65.3% |
| Combined | 9,394 | 35.7% | 37.6% |

Combining the two datasets resulted in a mean turning angle of just 0.0006 degrees, effectively now symmetric.

# Balancing

Even still, there was another bias that was present, which was driving in a straight line, with a driving angle of 0 degrees. Similar to the diabetes problem we examined in class, if we leave the data like this, a network which completely ignores the image and always outputs a steering angle of 0 would seemingly appear to be a success, despite being completely incorrect. To correct this, we divided out data into bins of steering ranges, and limited each bin to only 400 samples, randomly discarding down to 400 if they exceed it. This resulted in reducing the sample size of 9364 down to 2762, only 30% of the actual full data set.

[TODO] ATTACH IMAGE OF GRAPHS

# Pre-processing

During pre-processing, images are loaded from disk and converted from BGR to RGB, then processed. The pre-processing function operates as follows: The image is then cropped so that only the road area remains, removing unessential elements such as the sky. Then, the image is converted to the YUV color space and resized to 200x66 pixels, the same specs as used by the model. 

To verify that this pipeline works as expected, a sample size of 3 images is taken to display a 3x3 grid of images. The grid has three columns: a column of original images, a column of cropped road sections, and a column of processed images. The final check confirms that the array matches the expected input shape before training. 



# Augmentation

Augmentation is a way of improving the generalization of a model, by increasing the diversity in a data set without adding new data. For example, you can apply techniques such as flipping, brightness adjustment, zooming, etc. We don't need augmentation in validation because at that stage we are simply measuring the accuracy of the model, not training it further. As well, during validation we want to test the model on real world data, and artificially modifying the images defeats the purpose. 

# Model Architecture

<img width="351" height="502" alt="image" src="https://github.com/user-attachments/assets/f1a40ffa-e8c7-4f7f-8c80-199e6c240715" />
The model architecture consists of five convolutional layers followed by three fully-connected dense layers. The first three convolutional layers use 5x5 kernels with a (2, 2) stride to downsample the image, followed by two unstrided convolutional layers with 3x3 kernels to capture higher-level spatial details. The resulting feature map is then flattened into 1,164 neurons that are passed through dense layers of 100,50, and 10 neurons, culminating in a final 1-neuron output for vehicle control.

# Training

The model was trained using the Mean Squared Error loss function and the Adam optimizer with a learning rate of 0.0001. Training was conducted over 15 epochs with 300 steps per epoch and a batch size of 100. 
During training, the batches were streamed dynamically by the batch generator, applying augmentation on the data set before passing images through the preprocessing pipeline. 
Conversely, the validation set was built using unaugmented preprocessed images to evaluate the model's performance in real world conditions.

The loss curve rapidly decreases in the first three epochs as the model learns basic spatial features such as road borders and lane positions. From epoch 4 onward, the training loss decreases at a more steady pace, eventually flattening out at an error of 0.0062. 
Validation loss remains lower than training loss consistently through the loss curve, with the biggest difference being in the first three epochs. This is likely due to the model having more trouble with the augmented images in the training data set than the unmodified images in the validation data set. 

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/0b9a42b9-0e53-4ef1-9cb5-41b22b97b294" />

# Running the Project

## Training

[TODO]

```
python.exe training.py
```

# Testing in the simulator

```
python TestSimulation.py
```

[TODO] And then open the simulator

# Results

[TODO]

# Challenges

[TODO]

## Colour space conversion

[TODO]

## Training time

[TODO]

