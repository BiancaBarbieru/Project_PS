import cv2
import numpy as np
from build_laplacian_pyramid import build_laplacian_pyramid
from pseudo_cross_bilateral_filter import pseudo_cross_bilateral_filter
from wiener_filter import adaptive_wiener_filter
from add_noise import add_noise


def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
    no_levels = len(laplacian_pyramid)

    # Start with the finest level
    reconstructed_image = laplacian_pyramid[-1]

    # Iterate from coarser to finer levels
    for level in range(no_levels - 2, -1, -1):
        # Upsample the current level using OpenCV
        upsampled_level = cv2.pyrUp(reconstructed_image)

        # Make sure dimensions match before adding
        if upsampled_level.shape != laplacian_pyramid[level].shape:
            upsampled_level = upsampled_level[:laplacian_pyramid[level].shape[0], :laplacian_pyramid[level].shape[1]]

        # Add the upsampled level to the reconstructed image
        reconstructed_image = cv2.add(upsampled_level, laplacian_pyramid[level])

        # Normalize the pixel values to stay within the valid range (0 to 255)
        reconstructed_image = np.clip(reconstructed_image, 0, 255)

    return np.uint8(reconstructed_image)


# Read the noised and wiener images
original_img = cv2.imread('./Lena_BW.jpg', cv2.IMREAD_GRAYSCALE)
noised_img = add_noise(original_img)
wiener_filtered_img = adaptive_wiener_filter(noised_img)

# Specify the number of levels in the Laplacian pyramid
no_levels = 4

# Build the Laplacian pyramid for the noised image
laplacian_pyramid_noised = build_laplacian_pyramid(noised_img, no_levels)

# Build the Laplacian pyramid for the wiener filtered image
laplacian_pyramid_wiener = build_laplacian_pyramid(wiener_filtered_img, no_levels)

# Apply the pseudo cross bilateral filter
sigma_spatial = 2.0
sigma_range = 20.0
filtered_pyramid = [pseudo_cross_bilateral_filter(level_noised, level_wiener, sigma_spatial, sigma_range)
                    for level_noised, level_wiener in zip(laplacian_pyramid_noised, laplacian_pyramid_wiener)]

# Reconstruct the final image from the pyramid layers after the pseudo cross bilateral filter
reconstructed_image = reconstruct_from_laplacian_pyramid(filtered_pyramid)

# Display and save the final image
cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.imwrite('./Reconstructed_Image.jpg', reconstructed_image)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
