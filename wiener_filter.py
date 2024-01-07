import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.ndimage import convolve
import cv2


def adaptive_wiener_filter(image, block_size=5):
    # Convert the image to float32 for filter operations
    image_float = np.float32(image)

    # Estimate local mean and variance
    local_mean = cv2.boxFilter(image_float, -1, (block_size, block_size), borderType=cv2.BORDER_REFLECT)
    local_var = cv2.boxFilter((image_float - local_mean) ** 2, -1, (block_size, block_size), borderType=cv2.BORDER_REFLECT)

    # Estimate the noise variance
    noise_var = np.mean(local_var)

    # Wiener filter formula
    wiener_filter = local_var / (local_var + noise_var)

    # Apply Wiener filter
    filtered_image = local_mean + wiener_filter * (image_float - local_mean)

    # Clip values to the valid intensity range
    filtered_image = np.clip(filtered_image, 0, 255)

    return np.uint8(filtered_image)


# Creating the filtered image from the noised one
observed_image = cv2.imread('./Noised_Lena_BW.jpg', cv2.IMREAD_GRAYSCALE)
filtered_image_adaptive = adaptive_wiener_filter(observed_image)

# Display and save the filtered image
cv2.imshow('Adaptive Wiener Filtered Image', filtered_image_adaptive)
cv2.imwrite('Wiener_Filter_Over_Noised_Lena.jpg', filtered_image_adaptive)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
