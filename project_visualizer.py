import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils  # Assuming utils contains your utility functions

# Use st.cache to cache computationally expensive functions
@st.cache_data
def process_data(dicom_zip_file, masks_zip_file):
    dicom_volume, voxel_dimensions = utils.load_dicom_volume_from_zip(dicom_zip_file)

    ground_truth_masks = None
    if masks_zip_file is not None:
        ground_truth_masks = utils.load_masks_from_zip(masks_zip_file)

    return dicom_volume, voxel_dimensions, ground_truth_masks

@st.cache_data
def preprocess_and_infer(dicom_volume, voxel_dimensions, ground_truth_masks):
    preprocessed_volume, preprocessed_masks = utils.preprocess_data(dicom_volume, voxel_dimensions, ground_truth_masks)
    model, inferred_masks = utils.model_inference(preprocessed_volume)

    return model, preprocessed_volume, inferred_masks, preprocessed_masks

@st.cache_data
def calculate_and_display_metrics(model, inferred_masks, preprocessed_masks, selected_slice):
    accuracy, mean_iou = utils.calculate_metrics(model, inferred_masks[selected_slice], preprocessed_masks[selected_slice])
    st.info(f'Accuracy: {accuracy * 100:.2f}% - Mean IoU: {mean_iou * 100:.2f}%')

def main():
    st.title("MRI Segmentation Visualizer")

    # Upload zipped DICOM volume and masks (optional)
    dicom_zip_file = st.file_uploader("Upload Zipped DICOM Volume", type=["zip"])
    masks_zip_file = st.file_uploader("Upload Zipped Masks (optional)", type=["zip"])

    # Check if DICOM zip file is uploaded
    if dicom_zip_file is not None:
        # Process data
        dicom_volume, voxel_dimensions, ground_truth_masks = process_data(dicom_zip_file, masks_zip_file)

        # Proceed only if a button is clicked
        #if st.button("Process"):
        # Preprocess data and perform inference
        model, preprocessed_volume, inferred_masks, preprocessed_masks = preprocess_and_infer(dicom_volume, voxel_dimensions, ground_truth_masks)

        # Display interactive slider for selecting slice
        selected_slice = st.slider("Select Slice", 0, inferred_masks.shape[0] - 1)

        # Display the selected slice with different overlays
        display_slice(preprocessed_volume, inferred_masks, preprocessed_masks, selected_slice)

        if ground_truth_masks is not None:
            # Calculate and display metrics
            calculate_and_display_metrics(model, inferred_masks, preprocessed_masks, selected_slice)

def display_slice(volume, inferred_masks, ground_truth_masks, selected_slice):
    # Get the slice from the volume, along with the corresponding masks
    image = volume[selected_slice]
    inferred_mask = inferred_masks[selected_slice]

    if ground_truth_masks is not None:
        ground_truth_mask = ground_truth_masks[selected_slice]

    # Define the overlay with the inferred mask
    inferred_mask_overlay = overlay_mask_on_image(image, inferred_mask)

    # Define the overlay with the ground truth mask (if provided)
    if ground_truth_masks is not None:
        ground_truth_mask_overlay = overlay_mask_on_image(image, ground_truth_mask)

    # Display the slice, then the slice with the inferred mask, then the slice with the ground truth mask (if provided) in a single row, using st.image
    if ground_truth_masks is not None:
        st.image([image, inferred_mask_overlay, ground_truth_mask_overlay], caption=['Image', 'Inferred Mask', 'Ground Truth Mask'], use_column_width=True)
    else:
        st.image([image, inferred_mask_overlay], caption=['Image', 'Inferred Mask'], use_column_width=True, clamp=True)

def overlay_mask_on_image(image, mask, overlay_percentage=0.5):
    # Convert the image to a format suitable for overlay
    image_overlay = (image * 255).astype(np.uint8)
    image_overlay = cv2.cvtColor(image_overlay, cv2.COLOR_GRAY2RGBA)

    # Convert the mask to an RGBA format with transparency
    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)

    # Define colors for each class (adjust as needed)
    class_colors = {
        0: (0, 0, 0, 0),  # Background (transparent)
        1: (255, 0, 0, 255),  # Class 1
        2: (0, 255, 0, 255),  # Class 2
        3: (0, 0, 255, 255),  # Class 3
        4: (255, 255, 0, 255)  # Class 4
    }

    for class_idx, color in class_colors.items():
        mask_rgba[mask == class_idx, :] = color

    # Blend the image and mask with adjustable overlay percentage
    overlay = cv2.addWeighted(image_overlay, 1 - overlay_percentage, mask_rgba, overlay_percentage, 0)
    overlay = np.where(mask_rgba == 0, image_overlay, overlay)

    return overlay

if __name__ == "__main__":
    main()
