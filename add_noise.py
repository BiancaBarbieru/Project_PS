import cv2
import numpy as np


def add_noise(img, noise_level = 0.2):
    # Calculate the standard deviation based on the specified noise level
    standard_deviation = noise_level * np.max(img)

    # Generate Gaussian noise with the calculated standard deviation
    noise = np.random.normal(0, standard_deviation, img.shape)

    # Add the generated noise to the original image
    img_noised = img + noise

    # Clip the values to ensure they are within the valid range [0, 255]
    img_noised = np.clip(img_noised, 0, 255)

    return np.uint8(img_noised)


original_img = cv2.imread('./Lena_BW.jpg', cv2.IMREAD_GRAYSCALE)

# Resize the image to have dimensions as powers of 2 (just to be sure)
original_size = (2**int(np.log2(original_img.shape[1])), 2**int(np.log2(original_img.shape[0])))
original_img = cv2.resize(original_img, original_size)

# Creating the noised image from the original one
img_noised = add_noise(original_img)

# Display the original and noised images
# cv2.imshow('Original image', original_img)
# cv2.imshow('Noised image', img_noised)

# Save the noised image to a file
cv2.imwrite('./Noised_Lena_BW.jpg', img_noised)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
