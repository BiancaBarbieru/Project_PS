import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.ndimage import convolve
import cv2

def adaptive_wiener_filter(image, block_size=3):
    # Local mean and variance
    local_mean = convolve(image, np.ones((block_size, block_size)) / (block_size ** 2), mode='reflect')
    local_var = convolve((image - local_mean) ** 2, np.ones((block_size, block_size)) / (block_size ** 2), mode='reflect')

    # Calculate SNR
    snr = local_mean / local_var

    # Adaptive Wiener filter formulation
    filter_response = snr / (snr + 1)

    # Apply adaptive Wiener filter in the frequency domain
    filtered_image = np.abs(ifft2(fft2(image) * filter_response))

    return filtered_image

# Example usage
observed_image = cv2.imread('./Noised_Lena_BW.jpg', cv2.IMREAD_GRAYSCALE)

filtered_image_adaptive = adaptive_wiener_filter(observed_image)

# Display the filtered image
cv2.imshow('Adaptive Wiener Filtered Image', np.uint8(filtered_image_adaptive))
cv2.imwrite('Wiener_Filter_Over_Noised_Lena.jpg', np.uint8(filtered_image_adaptive))
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
