####################################################################################################

# Imports and settings

####################################################################################################

import os
import zipfile
import shutil
import numpy as np
import cv2
import pydicom
import skimage.transform
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Fix randomness and hide warnings
SEED = 1234
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
np.random.seed(SEED)

import random
random.seed(SEED)

import logging


# Import tensorflow and related libraries
import tensorflow as tf
from tensorflow import keras
from keras import backend
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(SEED)
tf.compat.v1.set_random_seed(SEED)


####################################################################################################

# Global variables

####################################################################################################

# Variables for loading and preprocessing the data

PATIENTS_TRAIN = np.array([1,2,3,5,8,10,13,15,19,20,21,22,31,32,33,34,36,37,38,39])
PATIENTS_TEST = np.array([4,6,7,9,11,12,14,16,17,18,23,24,25,26,27,28,29,30,35,40])

DATA_TRAIN_PATH = 'data/Train_Sets/MR/'
DATA_TEST_PATH = 'data/Test_Sets/MR/'
T1_SAMPLES_PATH = '/T1DUAL/DICOM_anon/InPhase/'
T1_LABELS_PATH = '/T1DUAL/Ground/'
T2_SAMPLES_PATH = '/T2SPIR/DICOM_anon/'
T2_LABELS_PATH = '/T2SPIR/Ground/'
DCM_EXT = '.dcm'
PNG_EXT = '.png'

PREPROCESSED_DATA_TRAIN_PATH = 'data/train/'
PREPROCESSED_DATA_TEST_PATH = 'data/test/'

TARGET_RESOLUTION = [1, 1, 1]
AVERAGE_VOLUME_SHAPE = [32, 272, 272]


# Variables for preparing the data for training

LIVER_RANGE = [55, 70]
RIGHT_KIDNEY_RANGE = [110, 135]
LEFT_KIDNEY_RANGE = [175, 200]
SPLEEN_RANGE = [240, 255]

LIVER_VALUE = 63
RIGHT_KIDNEY_VALUE = 126
LEFT_KIDNEY_VALUE = 189
SPLEEN_VALUE = 252

CATEGORY_MAP = {
    0: 0,   # Background
    63: 1,  # Liver
    126: 2, # Right kidney
    189: 3, # Left kidney
    252: 4  # Spleen
}


# Variables for training the network
SEED = 1234


# Variables for plotting
SAMPLE_FIGSIZE = (10, 10)
RESULTS_FIGSIZE = (18, 3)
PREDICTION_FIGSIZE = (6,18)
DEFAULT_CMAP = 'gray'
DEFAULT_LABEL_CMAP = 'jet'


# Variables for streamlit visualizer
TEMP_DIR = 'temp_extracted_folder'
MODEL_PATH = 'models/'
BEST_MODEL = 'unet_da_weights.keras'
NUM_CLASSES = len(CATEGORY_MAP)
BATCH_SIZE = 8

@keras.utils.register_keras_serializable()
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=None, name="mean_iou", dtype=None, **kwargs):
        super(UpdatedMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


####################################################################################################

# Public functions

####################################################################################################

# Getters

def get_patients_train():
    return PATIENTS_TRAIN

def get_patients_test():
    return PATIENTS_TEST

def get_liver_range():
    return LIVER_RANGE

def get_right_kidney_range():
    return RIGHT_KIDNEY_RANGE

def get_left_kidney_range():
    return LEFT_KIDNEY_RANGE

def get_spleen_range():
    return SPLEEN_RANGE

def get_liver_value():
    return LIVER_VALUE

def get_right_kidney_value():
    return RIGHT_KIDNEY_VALUE

def get_left_kidney_value():
    return LEFT_KIDNEY_VALUE

def get_spleen_value():
    return SPLEEN_VALUE

def get_category_map():
    return CATEGORY_MAP


# Functions for loading and preprocessing the data

def load_data(mode='train', verbose=True):
    if mode not in ['train', 'test']:
        raise ValueError("mode must be either 'train' or 'test'")
    
    if mode == 'train':
        data_path = DATA_TRAIN_PATH
        patients = PATIENTS_TRAIN
    else:
        data_path = DATA_TEST_PATH
        patients = PATIENTS_TEST
    
    # Load T1DUAL InPhase and T2SPIR samples separately
    T1_samples_volumes, T1_samples_voxel_dimensions, T2_samples_volumes, T2_samples_voxel_dimensions = _load_samples(patients, data_path, verbose=verbose)

    # Compute the average volume shape
    average_volume_shape = _compute_average_volume_shape(mode, T1_samples_volumes, T2_samples_volumes, verbose=verbose)

    # Preprocess samples
    T1_samples, T2_samples = _preprocess_samples(patients, T1_samples_volumes, T1_samples_voxel_dimensions, T2_samples_volumes, T2_samples_voxel_dimensions, average_volume_shape, verbose=verbose)

    if mode == 'train':
        # Load T1DUAL and T2SPIR labels separately
        T1_labels_volumes, T2_labels_volumes = _load_labels(patients, data_path, verbose=verbose)

        # Preprocess labels
        T1_labels, T2_labels = _preprocess_labels(patients, T1_labels_volumes, T1_samples_voxel_dimensions, T2_labels_volumes, T2_samples_voxel_dimensions, average_volume_shape, verbose=verbose)
    else:
        T1_labels = None
        T2_labels = None

    # Return the preprocessed images and labels
    return T1_samples, T2_samples, T1_labels, T2_labels

def load_preprocessed_data_train():
    # Load T1DUAL and T2SPIR samples separately
    T1_samples = np.load(f'{PREPROCESSED_DATA_TRAIN_PATH}T1_samples_train.npy', allow_pickle=True)
    T2_samples = np.load(f'{PREPROCESSED_DATA_TRAIN_PATH}T2_samples_train.npy', allow_pickle=True)

    # Load T1DUAL and T2SPIR labels separately
    T1_labels = np.load(f'{PREPROCESSED_DATA_TRAIN_PATH}T1_labels_train.npy', allow_pickle=True)
    T2_labels = np.load(f'{PREPROCESSED_DATA_TRAIN_PATH}T2_labels_train.npy', allow_pickle=True)

    # Transform the loaded data back into dictionaries
    T1_samples = T1_samples.item()
    T2_samples = T2_samples.item()
    T1_labels = T1_labels.item()
    T2_labels = T2_labels.item()

    # Return the preprocessed images and labels
    return T1_samples, T2_samples, T1_labels, T2_labels

def load_preprocessed_data_test():
    # Load T1DUAL and T2SPIR samples separately
    T1_samples = np.load(f'{PREPROCESSED_DATA_TEST_PATH}T1_samples_test.npy', allow_pickle=True)
    T2_samples = np.load(f'{PREPROCESSED_DATA_TEST_PATH}T2_samples_test.npy', allow_pickle=True)

    # Transform the loaded data back into dictionaries
    T1_samples = T1_samples.item()
    T2_samples = T2_samples.item()

    # Return the preprocessed images
    return T1_samples, T2_samples


# Functions for preparing the data for training

def prepare_data_for_training(T1_samples, T2_samples, T1_labels, T2_labels, verbose=True):
    if verbose:
        print('Preparing T1 samples for training...')

    T1_data_samples = _prepare_samples_for_network(T1_samples, verbose=verbose)

    if verbose:
        print('\nPreparing T2 samples for training...')

    T2_data_samples = _prepare_samples_for_network(T2_samples, verbose=verbose)

    if verbose:
        print('\nPreparing T1 labels for training...')

    T1_data_labels = _prepare_labels_for_network(T1_labels, verbose=verbose)

    if verbose:
        print('\nPreparing T2 labels for training...')

    T2_data_labels = _prepare_labels_for_network(T2_labels, verbose=verbose)

    return T1_data_samples, T2_data_samples, T1_data_labels, T2_data_labels

def prepare_data_for_testing(T1_samples, T2_samples, verbose=True):
    if verbose:
        print('Preparing T1 samples for testing...')

    T1_data_samples = _prepare_samples_for_network(T1_samples, verbose=verbose)

    if verbose:
        print('\nPreparing T2 samples for testing...')

    T2_data_samples = _prepare_samples_for_network(T2_samples, verbose=verbose)

    return T1_data_samples, T2_data_samples

def split_data_for_training(data_samples, data_labels, val_split_size=200, test_split_size=200, seed=SEED, verbose=True):
    if verbose:
        print('Splitting the data into training, validation and test sets...')

    # Split the data into training, validation and test sets
    train_val_samples, test_samples, train_val_labels, test_labels = train_test_split(data_samples, data_labels, test_size=test_split_size, random_state=seed)
    train_samples, val_samples, train_labels, val_labels = train_test_split(train_val_samples, train_val_labels, test_size=val_split_size, random_state=seed)

    if verbose:
        print(f'    Train samples shape: {train_samples.shape}')
        print(f'    Train labels shape: {train_labels.shape}')
        print(f'    Validation samples shape: {val_samples.shape}')
        print(f'    Validation labels shape: {val_labels.shape}')
        print(f'    Test samples shape: {test_samples.shape}')
        print(f'    Test labels shape: {test_labels.shape}')

    return train_samples, val_samples, test_samples, train_labels, val_labels, test_labels
    

# Functions for training the network

def get_unet_model(input_shape, num_classes, seed=SEED):
    # Set the random seed
    tf.random.set_seed(seed)

    # Input layer
    tf.random.set_seed(seed)
    input_layer = keras.layers.Input(shape=input_shape, name='input_layer')

    # First downsampling block
    down_sampling_block_1 = _unet_block(input_layer, 64, name='down_block1_')
    d1 = keras.layers.MaxPooling2D()(down_sampling_block_1)

    # Second downsampling block
    down_sampling_block_2 = _unet_block(d1, 128, name='down_block2_')
    d2 = keras.layers.MaxPooling2D()(down_sampling_block_2)
    
    # Third downsampling block
    down_sampling_block_3 = _unet_block(d2, 256, name='down_block3_')
    d3 = keras.layers.MaxPooling2D()(down_sampling_block_3)

    # Forth downsampling block
    down_sampling_block_4 = _unet_block(d3, 512, name='down_block4_')
    d4 = keras.layers.MaxPooling2D()(down_sampling_block_4)

    # Bottleneck
    bottleneck = _unet_block(d4, 512, name='bottleneck')

    # First upsampling block
    u1 = keras.layers.UpSampling2D(interpolation='bilinear')(bottleneck)
    u1 = keras.layers.Concatenate(name='concatenate1')([u1, down_sampling_block_4])
    u1 = _unet_block(u1, 256, name='up_block1_')

    # Second upsampling block
    u2 = keras.layers.UpSampling2D(interpolation='bilinear')(u1)
    u2 = keras.layers.Concatenate(name='concatenate2')([u2, down_sampling_block_3])
    u2 = _unet_block(u2, 128, name='up_block2_')

    # Third upsampling block
    u3 = keras.layers.UpSampling2D(interpolation='bilinear')(u2)
    u3 = keras.layers.Concatenate(name='concatenate3')([u3, down_sampling_block_2])
    u3 = _unet_block(u3, 64, name='up_block3_')

    # Forth upsampling block
    u4 = keras.layers.UpSampling2D(interpolation='bilinear')(u3)
    u4 = keras.layers.Concatenate(name='concatenate4')([u4, down_sampling_block_1])
    u4 = _unet_block(u4, 64, name='up_block4_')

    # Output layer
    output_layer = keras.layers.Conv2D(num_classes, kernel_size=3, padding='same', activation='softmax', name='output_layer')(u4)

    # Create the model
    model = keras.models.Model(inputs=input_layer, outputs=output_layer, name='unet_model')

    return model

def _unet_block(input_layer, num_filters, kernel_size=3, activation='relu', stack_size=2, name=''):
    x = input_layer

    for i in range(stack_size):
        x = keras.layers.Conv2D(num_filters, kernel_size, padding='same', name=f'{name}conv{i+1}')(x)
        x = keras.layers.BatchNormalization(name=f'{name}bn{i+1}')(x)
        x = keras.layers.Activation(activation, name=f'{name}act{i+1}')(x)

    return x


# Functions for plotting

def plot_sample(sample, label=None, plot_separately=False, title=None, figsize=SAMPLE_FIGSIZE, cmap=DEFAULT_CMAP, label_cmap=DEFAULT_LABEL_CMAP):
    if plot_separately and label is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(sample, cmap=cmap)
        axes[0].set_title('Sample')
        axes[0].axis('off')
        axes[1].imshow(label, cmap=label_cmap)
        axes[1].set_title('Label')
        axes[1].axis('off')
    elif label is not None:
        plt.figure(figsize=figsize)
        plt.imshow(sample, cmap=cmap)
        plt.imshow(label, cmap=label_cmap, alpha=0.5)
        if title is not None:
            plt.title(title)
    else:
        plt.figure(figsize=figsize)
        plt.imshow(sample, cmap=cmap)
        if title is not None:
            plt.title(title)
    
    plt.axis('off')
    plt.show()

def plot_results(history):
    # Get the best epoch
    best_epoch = np.argmax(history.history['val_mean_iou'])

    # Plot the loss (train and validation)
    plt.figure(figsize=RESULTS_FIGSIZE)
    plt.plot(history.history['loss'], label='Training', alpha=.8, color='#ff7f0e', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', alpha=.9, color='#5a9aa5', linewidth=2)
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Cross Entropy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    # Plot the accuracy (train and validation)
    plt.figure(figsize=RESULTS_FIGSIZE)
    plt.plot(history.history['accuracy'], label='Training', alpha=.8, color='#ff7f0e', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', alpha=.9, color='#5a9aa5', linewidth=2)
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    # Plot the mean intersection over union (train and validation)
    plt.figure(figsize=RESULTS_FIGSIZE)
    plt.plot(history.history['mean_iou'], label='Training', alpha=.8, color='#ff7f0e', linewidth=2)
    plt.plot(history.history['val_mean_iou'], label='Validation', alpha=.9, color='#5a9aa5', linewidth=2)
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Mean Intersection over Union')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

def plot_prediction(model, sample, label, figsize=PREDICTION_FIGSIZE, cmap=DEFAULT_CMAP, label_cmap=DEFAULT_LABEL_CMAP):
    # Predict the sample
    prediction = model.predict(np.expand_dims(sample, axis=0))[0]
    prediction = np.argmax(prediction, axis=-1)

    # Plot the sample alone, the sample with the label and the sample with the prediction
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(sample, cmap=cmap)
    axes[0].set_title('Sample')
    axes[0].axis('off')

    axes[1].imshow(sample, cmap=cmap)
    axes[1].imshow(label, cmap=label_cmap, alpha=0.5)
    axes[1].set_title('Label')
    axes[1].axis('off')

    axes[2].imshow(sample, cmap=cmap)
    axes[2].imshow(prediction, cmap=label_cmap, alpha=0.5)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.show()


# Functions for streamlit visualizer

def load_dicom_volume_from_zip(zip_file):
    # Extract the contents of the zip file to a temporary directory
    temp_dir = TEMP_DIR
    os.makedirs(temp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

        # Define the path to the extracted folder
        extracted_folder = os.path.join(temp_dir, zip_file.name.split('.')[0])

    # Load DICOM images from the extracted folder
    volume, voxel_dimensions = _visualizer_load_volume(extracted_folder)

    # Cleanup: Remove the temporary extracted folder
    shutil.rmtree(temp_dir)

    return volume, voxel_dimensions

def load_masks_from_zip(zip_file):
    # Extract the contents of the zip file to a temporary directory
    temp_dir = TEMP_DIR
    os.makedirs(temp_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Load masks from the extracted folder
    masks = _visualizer_load_masks(temp_dir, verbose=False)

    # Cleanup: Remove the temporary extracted folder
    shutil.rmtree(temp_dir)

    return masks

def preprocess_data(dicom_volume, voxel_dimensions, masks=None):
    # Preprocess the volume
    preprocessed_volume = _preprocess_volume(dicom_volume, AVERAGE_VOLUME_SHAPE, voxel_dimensions)

    # If a segmentation is provided, preprocess it as well
    preprocessed_masks = None
    if masks is not None:
        preprocessed_masks = _preprocess_volume(masks, AVERAGE_VOLUME_SHAPE, voxel_dimensions)
    
    # Normalize the volume with min-max scaling
    preprocessed_volume = (preprocessed_volume - np.min(preprocessed_volume)) / (np.max(preprocessed_volume) - np.min(preprocessed_volume))

    return preprocessed_volume, preprocessed_masks

def model_inference(preprocessed_volume, verbose=True):

    if verbose:
        print('Performing inference...')

    # Reshape the inputs for the model
    preprocessed_volume = np.expand_dims(preprocessed_volume, axis=-1)

    # Create the model and load the weights
    model = get_unet_model(preprocessed_volume.shape[1:], NUM_CLASSES)
    model.load_weights(f'{MODEL_PATH}{BEST_MODEL}')

    if verbose:
        print(f'- Model input shape: {model.input_shape}')
        print(f'- Model output shape: {model.output_shape}')

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', UpdatedMeanIoU(num_classes=NUM_CLASSES)])

    # Perform inference
    predictions = model.predict(preprocessed_volume, batch_size=BATCH_SIZE)
    predictions = np.argmax(predictions, axis=-1)

    if verbose:
        print(f'Predictions shape: {predictions.shape}')

    return model, predictions

def calculate_metrics(model, inferred_mask, ground_truth_mask):
    # Evaluate the model
    _, accuracy, mean_iou = model.evaluate(inferred_mask, ground_truth_mask)

    return accuracy, mean_iou

####################################################################################################

# Private functions

####################################################################################################

# Functions for loading and preprocessing the data

def _load_samples(patients, data_path, verbose=True):
    if verbose:
        print('Loading T1 data...')

    T1_samples_volumes, T1_samples_voxel_dimensions = _load_volumes(patients, data_path, T1_SAMPLES_PATH, verbose=verbose)

    if verbose:
        print('\nLoading T2 data...')

    T2_samples_volumes, T2_samples_voxel_dimensions = _load_volumes(patients, data_path, T2_SAMPLES_PATH, verbose=verbose)

    return T1_samples_volumes, T1_samples_voxel_dimensions, T2_samples_volumes, T2_samples_voxel_dimensions

def _load_labels(patients, data_path, verbose=True):
    if verbose:
        print('\nLoading T1 labels...')
    
    T1_labels_volumes = _load_images(patients, data_path, T1_LABELS_PATH, verbose=verbose)

    if verbose:
        print('\nLoading T2 labels...')
    
    T2_labels_volumes = _load_images(patients, data_path, T2_LABELS_PATH, verbose=verbose)

    return T1_labels_volumes, T2_labels_volumes

def _load_volumes(patients, data_path, samples_path, verbose=True):
    # Save each patient's volume in a dictionary with the patient's index as the key
    image_volumes = {}

    # Create a didctionaru of the voxel dimensions with the patient's index as the key
    voxel_dimensions = {}

    for patient_index in patients:

        if verbose:
            print(f'- Loading image volume for patient {patient_index}...')

        image_vol = []
        
        # Load an individual volume
        for _,  _, files in sorted(os.walk(f'{data_path}{patient_index}{samples_path}')):
            for filename in (sorted(files)):
                if filename.endswith(DCM_EXT):
                    image_dcm_std = pydicom.dcmread(os.path.join(f'{data_path}{patient_index}{samples_path}', filename))

                    # Extract the image from the DICOM file
                    img = image_dcm_std.pixel_array
                    image_vol.append(img)
                
        x_space = image_dcm_std.PixelSpacing[0]
        y_space = image_dcm_std.PixelSpacing[1]
        z_space = image_dcm_std.SpacingBetweenSlices
        voxel_dimension = (x_space, y_space, z_space)

        image_volume_raw = np.array(image_vol)
        voxel_dimension = np.array(voxel_dimension)

        if verbose:
            print(f'    Image volume shape: {image_volume_raw.shape}')
            print(f'    Voxel dimensions: {voxel_dimension}')

        # Add the image volume to the dictionary of image volumes
        image_volumes[patient_index] = image_volume_raw
        
        # Add the voxel dimensions to the dictionary of voxel dimensions
        voxel_dimensions[patient_index] = voxel_dimension

    if verbose:
        print(f'\nLoaded {len(image_volumes)} patient image volumes\n')
    
    return image_volumes, voxel_dimensions

def _load_images(patients, data_path, labels_path, verbose=True):
    # Save each patient's target volume in a dictionary with the patient's index as the key
    target_volumes = {}

    for patient_index in patients:

        if verbose:
            print(f'- Loading target volume for patient {patient_index}...')

        target_vol = []
        
        # Load an individual volume
        for _,  _, files in sorted(os.walk(f'{data_path}{patient_index}{labels_path}')):
            for filename in (sorted(files)):
                if filename.endswith(PNG_EXT):
                    image = cv2.imread(os.path.join(f'{data_path}{patient_index}{labels_path}', filename), cv2.IMREAD_GRAYSCALE)
                    target_vol.append(image)

        target_volume_raw = np.array(target_vol)

        if verbose:
            print(f'    Target volume shape: {target_volume_raw.shape}')

        # Add the target volume to the dictionary of target volumes
        target_volumes[patient_index] = target_volume_raw

    if verbose:
        print(f'\nLoaded {len(target_volumes)} patient target volumes\n')

    return target_volumes

def _preprocess_samples(patients, T1_samples_volumes, T1_samples_voxel_dimensions, T2_samples_volumes, T2_samples_voxel_dimensions, average_volume_shape, verbose=True):
    if verbose:
        print('\nPreprocessing T1 images...')
    
    T1_samples = _preprocess_volumes(patients, T1_samples_volumes, average_volume_shape, T1_samples_voxel_dimensions, verbose=verbose)

    if verbose:
        print('\nPreprocessing T2 images...')
    
    T2_samples = _preprocess_volumes(patients, T2_samples_volumes, average_volume_shape, T2_samples_voxel_dimensions, verbose=verbose)

    return T1_samples, T2_samples

def _preprocess_volumes(patients, image_volumes, average_volume_shape, voxel_dimensions, verbose=True):
    # Create a dictionary of the preprocessed image volumes with the patient's index as the key
    preprocessed_image_volumes = {}

    # Preprocess the image volumes so that they are all the same shape, namely the average volume shape
    for patient_id in patients:
        if verbose:
            print(f'- Preprocessing patient {patient_id}...')

        # Get patient's image volume and voxel dimensions
        image_volume = image_volumes[patient_id]
        voxel_dimension = voxel_dimensions[patient_id]

        # Call a function to preprocess a single volume, passing the single volume and voxel dimensions
        preprocessed_image_volume = _preprocess_volume(image_volume, average_volume_shape, voxel_dimension, verbose=verbose)

        # Add the preprocessed image volume to the dictionary of preprocessed image volumes
        preprocessed_image_volumes[patient_id] = preprocessed_image_volume
        
    if verbose:
        print(f'\nPreprocessed {len(preprocessed_image_volumes)} patient image volumes')
    
    return preprocessed_image_volumes

def _preprocess_volume(image_volume, average_volume_shape, voxel_dimension, verbose=True):
    # Resample the volume to the target resolution
    scale_vector = np.asarray(voxel_dimension) / np.asarray(TARGET_RESOLUTION)
    resampled_volume = skimage.transform.rescale(image_volume, scale_vector, order=3, preserve_range=True, mode='constant')

    # Reshape the interpolated volume to the target shape
    factors = (np.asarray(average_volume_shape) / np.asarray(resampled_volume.shape))
    reshaped_volume = zoom(resampled_volume, factors, order=3, mode='nearest')
    
    if verbose:
        print(f'    Shape before preprocessing: {image_volume.shape}')
        print(f'    Shape after preprocessing: {reshaped_volume.shape}')
    
    return reshaped_volume

def _preprocess_labels(patients, T1_labels_volumes, T1_samples_voxel_dimensions, T2_labels_volumes, T2_samples_voxel_dimensions, average_volume_shape, verbose=True):
    if verbose:
        print('\nPreprocessing T1 labels...')

    T1_labels = _preprocess_volumes(patients, T1_labels_volumes, average_volume_shape, T1_samples_voxel_dimensions, verbose=verbose)

    if verbose:
        print('\nPreprocessing T2 labels...')

    T2_labels = _preprocess_volumes(patients, T2_labels_volumes, average_volume_shape, T2_samples_voxel_dimensions, verbose=verbose)

    return T1_labels, T2_labels

def _compute_average_volume_shape(mode, T1_samples_volumes, T2_samples_volumes, verbose=True):
    if mode == 'train':
        
        if verbose:
            print('\nComputing average volume shape...')
        
        T1_average_volume_shape = _get_average_volume_shape(T1_samples_volumes)
        T2_average_volume_shape = _get_average_volume_shape(T2_samples_volumes)
        volume_shapes = [T1_average_volume_shape, T2_average_volume_shape]
        average_volume_shape = np.round(np.mean(volume_shapes, axis=0)).astype(int)

        if verbose:
            print(f'\nAverage volume shape: {average_volume_shape}')
    else:
        # Use the average volume shape from the training set
        average_volume_shape = np.array(AVERAGE_VOLUME_SHAPE)

        if verbose:
            print(f'\nUsing average volume shape: {average_volume_shape}')
    
    return average_volume_shape

def _get_average_volume_shape(image_volumes):
    # Get the individual volume shapes
    volume_shapes = [volume.shape for volume in image_volumes.values()]
    
    # Compute the average volume shape, rounded to the nearest integer
    average_volume_shape = np.round(np.mean(volume_shapes, axis=0)).astype(int)

    # Return the average volume shape
    return average_volume_shape

# Functions for preparing the data for the network

def _prepare_samples_for_network(samples, verbose=True):
    if verbose:
        print('- Vertically stacking the samples from all patients...')

    # Vertically stack the samples from all patients
    data_samples = np.vstack([samples[patient_index] for patient_index in samples.keys()])

    if verbose:
        print(f'- Adding a channel dimension to the samples...')

    # Add a channel dimension to the samples
    data_samples = np.expand_dims(data_samples, axis=-1)

    if verbose:
        print(f'- Normalizing the samples with min-max scaling...')

    # Normalize the samples with min-max scaling
    data_samples = (data_samples - np.min(data_samples)) / (np.max(data_samples) - np.min(data_samples))

    if verbose:
        print(f'Samples shape: {data_samples.shape}')

    return data_samples

def _prepare_labels_for_network(labels, verbose=True):
    if verbose:
        print('- Vertically stacking the labels from all patients...')

    # Vertically stack the labels from all patients
    data_labels = np.vstack([labels[patient_index] for patient_index in labels.keys()])

    if verbose:
        print(f'- Creating masks for the different categories...')

    # Get the masks for the different categories
    liver_masks = np.where((data_labels >= LIVER_RANGE[0]) & (data_labels <= LIVER_RANGE[1]), LIVER_VALUE, 0)
    right_kidney_masks = np.where((data_labels >= RIGHT_KIDNEY_RANGE[0]) & (data_labels <= RIGHT_KIDNEY_RANGE[1]), RIGHT_KIDNEY_VALUE, 0)
    left_kidney_masks = np.where((data_labels >= LEFT_KIDNEY_RANGE[0]) & (data_labels <= LEFT_KIDNEY_RANGE[1]), LEFT_KIDNEY_VALUE, 0)
    spleen_masks = np.where((data_labels >= SPLEEN_RANGE[0]) & (data_labels <= SPLEEN_RANGE[1]), SPLEEN_VALUE, 0)

    # Combine the masks into a single mask
    data_labels = liver_masks + right_kidney_masks + left_kidney_masks + spleen_masks

    # Map the labels to the categories
    data_labels = np.vectorize(CATEGORY_MAP.get)(data_labels)

    # Add a channel dimension to the labels
    data_labels = np.expand_dims(data_labels, axis=-1)

    if verbose:
        print(f'Labels shape: {data_labels.shape}')

    return data_labels

# Functions for streamlit visualizer TODO

def _visualizer_load_volume(path, verbose=True):
    # Load the volumes present in the folder at the specified path
    volume = []

    valid_file = False

    for _,  _, files in sorted(os.walk(path)):
        for filename in (sorted(files)):
            if filename.endswith(DCM_EXT):
                valid_file = True

                image_dcm_std = pydicom.dcmread(os.path.join(path, filename))

                # Extract the image from the DICOM file
                img = image_dcm_std.pixel_array
                volume.append(img)
            else:
                valid_file = False

        if valid_file:
            x_space = image_dcm_std.PixelSpacing[0]
            y_space = image_dcm_std.PixelSpacing[1]
            z_space = image_dcm_std.SpacingBetweenSlices
            voxel_dimension = (x_space, y_space, z_space)

        image_volume_raw = np.array(volume)
        voxel_dimension = np.array(voxel_dimension)

        if verbose:
            print(f'Volume shape: {image_volume_raw.shape}')
            print(f'Voxel dimensions: {voxel_dimension}')

    return image_volume_raw, voxel_dimension

def _visualizer_load_masks(path, verbose=True):
    # Load the masks present in the folder at the specified path
    masks = []

    for _,  _, files in sorted(os.walk(path)):
        for filename in (sorted(files)):
            if filename.endswith(PNG_EXT):
                image = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
                masks.append(image)

    masks = np.array(masks)

    if verbose:
        print(f'Masks shape: {masks.shape}')

    return masks