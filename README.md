# Biomedical Computer Vision Project

## Description

This project is part of the Biomedical Computer Vision course at the Polytechnic University of Milan.
It focuses on computer vision in the medical field, specifically addressing organ segmentation in MRI scans.
The goal is to train a deep learning model capable of segmenting the liver, two kidneys, and spleen in each slice of DICOM volumes obtained from the CHAOS 2019 dataset.

## Dataset

The CHAOS 2019 dataset is utilized for this project. It includes MRI scans in DICOM format for 40 patients, each with corresponding ground truth segmentations.
The used part of the dataset is divided into two subsets: T1DUAL-InPhase and T2SPIR, each comprising 20 patients. The ground truth segmentations enable supervised training of the deep learning model.

## Model/Models

<!-- Add a detailed description of the model or models used in this project. -->

## Results Comparison

<!-- Discuss and compare the results obtained with different models. -->

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