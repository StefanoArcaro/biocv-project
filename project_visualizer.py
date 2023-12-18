import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils

# Set page configuration
st.set_page_config(
    page_title="MRI Segmentation Visualizer",
    page_icon=":brain:",
    layout="centered"
)

@st.cache_data
def load_data(dicom_zip_file, masks_zip_file):
    dicom_volume, voxel_dimensions = utils.load_dicom_volume_from_zip(dicom_zip_file)

    ground_truth_masks = None
    if masks_zip_file is not None:
        ground_truth_masks = utils.load_masks_from_zip(masks_zip_file)
        print(ground_truth_masks.shape)

    return dicom_volume, voxel_dimensions, ground_truth_masks

@st.cache_data
def preprocess_volume(dicom_volume, voxel_dimensions):
    preprocessed_volume = utils.preprocess_data_volume(dicom_volume, voxel_dimensions)

    return preprocessed_volume

@st.cache_data
def preprocess_ground_truth_masks(ground_truth_masks, voxel_dimensions):
    preprocessed_masks = utils.preprocess_data_ground_truth_masks(ground_truth_masks, voxel_dimensions)

    return preprocessed_masks

@ st.cache_resource
def model_inference(preprocessed_volume):
    model, inferred_masks = utils.model_inference(preprocessed_volume, verbose=False)

    return model, inferred_masks

@st.cache_resource
def calculate_metrics(_model, inferred_masks, preprocessed_masks):
    accuracy_list, mean_iou_list, average_accuracy, average_mean_iou = utils.calculate_metrics(_model, inferred_masks, preprocessed_masks)

    return accuracy_list, mean_iou_list, average_accuracy, average_mean_iou

def display_metrics(accuracy_list, mean_iou_list, average_accuracy, average_mean_iou, selected_slice):
    # Display the average metrics, then the metrics for the selected slice
    st.info(f'Average Accuracy: {average_accuracy * 100:.2f}% - Average Mean IoU: {average_mean_iou * 100:.2f}%')
    st.info(f'Slice Accuracy: {accuracy_list[selected_slice] * 100:.2f}% - Slice Mean IoU: {mean_iou_list[selected_slice] * 100:.2f}%')

def display_slice(volume, inferred_masks, ground_truth_masks, selected_slice, overlay_percentage=0.5):
    # Get the slice from the volume, along with the corresponding masks
    image = volume[selected_slice]
    inferred_mask = inferred_masks[selected_slice]

    if ground_truth_masks is not None:
        ground_truth_mask = ground_truth_masks[selected_slice]

    # Define the overlay with the inferred mask
    inferred_mask_overlay = overlay_mask_on_image(image, inferred_mask, overlay_percentage)

    # Define the overlay with the ground truth mask (if provided)
    if ground_truth_masks is not None:
        ground_truth_mask_overlay = overlay_mask_on_image(image, ground_truth_mask, overlay_percentage)

    # Display the slice, then the slice with the inferred mask, then the slice with the ground truth mask (if provided) in a single row, using st.image
    if ground_truth_masks is not None:
        # Create three columns
        col1, col2, col3 = st.columns(3)

        # Display the slice in the first column
        col1.image(image, caption='Image', clamp=True)

        # Display the slice with the inferred mask in the second column
        col2.image(inferred_mask_overlay, caption='Inferred Mask', clamp=True)

        # Display the slice with the ground truth mask in the third column
        col3.image(ground_truth_mask_overlay, caption='Ground Truth Mask', clamp=True)
    else:
        # Create two columns
        col1, col2 = st.columns(2)

        # Display the slice in the first column
        col1.image(image, caption='Image', use_column_width=True, clamp=True)

        # Display the slice with the inferred mask in the second column
        col2.image(inferred_mask_overlay, caption='Inferred Mask', use_column_width=True, clamp=True)

def overlay_mask_on_image(image, mask, overlay_percentage=0.5):
    """
    Overlay a segmentation mask on an image with customizable colors and overlay opacity.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        mask (numpy.ndarray): Segmentation mask.
        overlay_colors (dict): Dictionary mapping class indices to RGBA color tuples.
                              Example: {0: (0, 0, 0, 0), 1: (255, 0, 0, 255), ...}
                              Default is None, which uses default colors.
        overlay_percentage (float): Overlay opacity (0.0 to 1.0).

    Returns:
        numpy.ndarray: Image with overlay.
    """
    # Create the color picker dictionary
    # overlay_colors = {
    #     0: "#000000",                                   # Background (transparent)
    #     1: st.color_picker("Liver", "#FF0000"),         # Class 1
    #     2: st.color_picker("Right Kidney", "#00FF00"),  # Class 2
    #     3: st.color_picker("Left Kidney", "#0000FF"),   # Class 3
    #     4: st.color_picker("Spleen", "#FFFF00")         # Class 4
    # }

    overlay_colors = {
        0: "#000000",   # Background (transparent)
        1: "#FF0000",   # Class 1
        2: "#00FF00",   # Class 2
        3: "#0000FF",   # Class 3
        4: "#FFFF00"    # Class 4
    }

    # Convert the image to a format suitable for overlay
    image_overlay = (image * 255).astype(np.uint8)
    image_overlay = cv2.cvtColor(image_overlay, cv2.COLOR_GRAY2RGBA)

    # Convert the mask to an RGBA format with transparency
    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)

    for class_idx, color in overlay_colors.items():
        # Convert hex to RGB
        rgb_color = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        rgba_color = (*rgb_color, 255)
        mask_rgba[mask == class_idx, :] = rgba_color

    # Blend the image and mask with adjustable overlay percentage
    overlay = cv2.addWeighted(image_overlay, 1 - overlay_percentage, mask_rgba, overlay_percentage, 0)
    overlay = np.where(mask_rgba == 0, image_overlay, overlay)

    return overlay

def main():
    st.title("MRI Organ Segmentation Visualizer")

    # Upload zipped DICOM volume and masks (optional)
    dicom_zip_file = st.file_uploader("Upload Zipped DICOM Volume", type=["zip"])
    masks_zip_file = st.file_uploader("Upload Zipped Masks (optional)", type=["zip"])

    # Check if DICOM zip file is uploaded
    if dicom_zip_file is not None:
        # Process data
        dicom_volume, voxel_dimensions, ground_truth_masks = load_data(dicom_zip_file, masks_zip_file)

        # Proceed only if a button is clicked
        #if st.button("Process"):
        # Preprocess data and perform inference
        # model, preprocessed_volume, inferred_masks, preprocessed_masks = preprocess_and_infer(dicom_volume, voxel_dimensions, ground_truth_masks)
        preprocessed_volume = preprocess_volume(dicom_volume, voxel_dimensions)

        preprocessed_masks = None
        if ground_truth_masks is not None:
            preprocessed_masks = preprocess_ground_truth_masks(ground_truth_masks, voxel_dimensions)
            
        model, inferred_masks = model_inference(preprocessed_volume)

        # Display interactive slider for selecting slice
        selected_slice = st.slider("Select Slice", 1, inferred_masks.shape[0]) - 1

        # Display interactive slider for selecting mask opacity
        overlay_percentage = st.slider("Select Mask Opacity", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        # Display the selected slice with different overlays
        display_slice(preprocessed_volume, inferred_masks, preprocessed_masks, selected_slice, overlay_percentage)

        if ground_truth_masks is not None:
            # Calculate and display metrics
            #calculate_and_display_metrics(model, inferred_masks, preprocessed_masks, selected_slice)
            accuracy_list, mean_iou_list, average_accuracy, average_mean_iou = calculate_metrics(model, inferred_masks, preprocessed_masks)
            display_metrics(accuracy_list, mean_iou_list, average_accuracy, average_mean_iou, selected_slice)

if __name__ == "__main__":
    main()
