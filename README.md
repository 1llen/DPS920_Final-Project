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

[TODO] EXPLAIN WHAT AUGMENTATION DOES AND WHY WE DO IT; ADD A TABLE IF NECESSARY. EXPLAIN THAT IT DOESN'T SHOW UP IN VALIDATION, ONLY TRAINING

Augmentation is a way of improving the generalization of a model, by increasing the diversity in a data set without adding new data. For example, you can apply techniques such as flipping, brightness adjustment, zooming, etc. We don't need augmentation in validation because at that stage we are simply measuring the accuracy of the model, not training it further. As well, during validation we want to test the model on real world data, and artificially modifying the images defeats the purpose. 

# Model Architecture

[TODO] SCREENSHOT AND PASTE THE DIAGRAM FROM THE INSTRUCTIONS, AND THEN BRIEFLY DESCRIBE THE PROCESS

# Training

[TODO] EXPLAIN THE PROCESS OF TRAINING (AND VALIDATION); EXPLAIN THE LOSS CURVE.

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

# Challenges

## Colour space conversion

## Training time

