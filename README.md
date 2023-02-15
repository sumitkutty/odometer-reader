# Odometer-Reader

## Usage:

#### Clone this Repository
```
> git clone git@github.com:sumitkutty/odometer-reader.git
```
#### Install Dependancies
```
> cd odometer-reader

> pip install -r requirements.txt
```

#### Run the test_predict.py on target folder
```
> python3 test_predict.py --images_path PATH_TO_IMAGE_FOLDER 

# To generate a csv file with OCR results:
> python3 test_predict.py --images_path PATH_TO_IMAGE_FOLDER --generate_csv
```

## Objective: To Detect and Recognize the digits in the images of odometer display


## Stages:
The project is required training two different models. One to detect the odometer display. Instance Segmentation (MaskRCNN) along with detectron2 as framework was used for this, and another to recognize the digits in the detected ROI which was implemented using a Scene-Text-Recognition open source framework.


## Detection
#### Dataset:
##### Size:
* Train: 3800
* Val: 200

##### Preparation
* The given dataset was split between multiple folders with each folder having its annotations files in the VIA JSON format.
* The dataset folders were merged along with its annotations.
* Created dataset dictionaries to facilitate detectron2 training.


##### Hyperparameters:
* Backbone: ResNet-50
* Batch Size: 4
* Classes: [LCD, odometer, M, not_touching, screen]

##### Results:
* Iterations Trained: 6000
* Total Loss: 0.5008

## Recognition
#### Dataset:
##### Size:
* Train: 3800
* Val: 200

##### Preparation
* The dataset created above was used to create a new dataset for the OCR.
* This involved writing scripts to crop the ground truth ROIs from the images and creating a new dataset.
* The labels for these were stored in a text file representing the annotations for the OCR dataset.


##### Hyperparameters
* Vocabulary: "0123456789."
* Batch Size: 256


##### Results:
* Iterations Trained: 3000
* Best Accuracy: 93.75
* Total Loss: 0.074
