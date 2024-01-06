import cv2
import numpy as np
from build_laplacian_pyramid import build_laplacian_pyramid
from denoising_per_levels import denoising_per_levels
from scipy.ndimage import zoom


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


original_img = cv2.imread('./Noised_Lena_BW.jpg', cv2.IMREAD_GRAYSCALE)

no_levels = 20

laplacian_pyramid = build_laplacian_pyramid(original_img, no_levels)
filtered_pyramid = [denoising_per_levels(level) for level in laplacian_pyramid]
reconstructed_image = reconstruct_from_laplacian_pyramid(filtered_pyramid)

cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.imwrite('./reconstructed_image.png', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
