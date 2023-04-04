import numpy as np
import cv2


def extract_n_largest_blobs(binary_mask, n=1, min_area=60):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the area of each contour
    areas = [cv2.contourArea(c) for c in contours]

    # Create a list of tuples with each contour and its corresponding area
    contour_list = list(zip(contours, areas))

    # Sort the contours in descending order of their areas
    sorted_contours = sorted(contour_list, key=lambda x: x[1], reverse=True)

    # Extract the n largest contours that have an area greater than min_area
    refined_mask = np.zeros_like(binary_mask)
    count = 0
    for cnt, area in sorted_contours:
        if area >= min_area:
            cv2.drawContours(refined_mask, [cnt], -1, 255, -1)
            count += 1
        if count >= n:
            break

    return refined_mask


def postprocessing_pipeline(original_image, reconstructed_image, threshold, kernel_size=(3, 3),
                            operations=['opening'], n_largest_blobs=3, blur_kernel_size=3, blur_sigma=1):
    # Convert images to the range [0, 255]
    original_image_255 = (original_image * 255).astype(np.uint8)
    reconstructed_image_255 = (reconstructed_image * 255).astype(np.uint8)

    # Compute the residual
    residual = cv2.absdiff(original_image_255, reconstructed_image_255)

    # Threshold the residual
    _, binary_mask = cv2.threshold(residual, threshold, 255, cv2.THRESH_BINARY)

    # Define the structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Apply morphological operations
    for operation in operations:
        if operation == 'erosion':
            binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
        elif operation == 'dilation':
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        elif operation == 'opening':
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        else:
            print(f"Warning: Unsupported operation '{operation}'")

    # Extract the n largest blobs
    refined_mask = extract_n_largest_blobs(binary_mask, n=n_largest_blobs)

    # Apply Gaussian blur
    refined_mask = cv2.GaussianBlur(refined_mask, (blur_kernel_size, blur_kernel_size), sigmaX=blur_sigma)

    return residual, binary_mask, refined_mask




