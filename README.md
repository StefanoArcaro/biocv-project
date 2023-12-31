# Biomedical Computer Vision Project

## Description

This project is part of the Biomedical Computer Vision course at the Polytechnic University of Milan.
It focuses on computer vision in the medical field, specifically addressing organ segmentation in MRI scans.
The goal is to train a deep learning model capable of segmenting the liver, two kidneys, and spleen in each slice of DICOM volumes obtained from the CHAOS 2019 dataset.

## Dataset

The CHAOS 2019 dataset is utilized for this project. It includes MRI scans in DICOM format for 40 patients, each with corresponding ground truth segmentations.
The used part of the dataset is divided into two subsets: T1DUAL-InPhase and T2SPIR, each comprising 20 patients. The ground truth segmentations enable supervised training of the deep learning model.

## Model/Models

The model used for this project is a U-Net, a convolutional neural network architecture that is commonly used for image segmentation tasks.
Two different data augmentation pipelines have also been used, although the results obtained with them are not satisfactory.

The UNet without data augmentation is therefore the best model for this project.

## Results Comparison

The results in the following table are expressed in terms of Mean Intersection over Union and are obtained by comparing the predicted segmentations with the ground truth segmentations.

|  | Unet - No Data Augmentation | UNet - Strong Data Augmentation | UNet - Weak Data Augmentation |
| :---         |     :---:      |          :---: |         :---: |
| **Training Set**   | <ins>0.8426     | 0.6158    | 0.8375      |
| **Test Set**     | <ins>0.8075       | 0.7919      | 0.8030    |

## Streamlit App

I also created a small web application with Streamlit to show the results of the model on any DICOM volume.

In this repository, you can also find a sample volume, together with its ground truth segmentations, to test the app.

To access the app, simply click on the following link: [https://segmentation-visualizer.streamlit.app/](https://segmentation-visualizer.streamlit.app/)

## How to Use

### Requirements

Ensure you have the required dependencies installed. You can set up the virtual environment using the provided `requirements.txt` file:

```bash
conda create --name your_virtual_environment_name python=3.11
conda activate your_virtual_environment_name
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License.