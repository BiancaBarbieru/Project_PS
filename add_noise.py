import cv2
import numpy as np

def add_noise(img, noise_level):
    standard_deviation = noise_level * np.max(img)

    noise = np.random.normal(0, standard_deviation, img.shape)

    img_noised = img + noise

    img_noised = np.clip(img_noised, 0, 255)

    return img_noised

original_img = cv2.imread('./Lena_BW.jpg', cv2.IMREAD_GRAYSCALE)
original_size = (2**int(np.log2(original_img.shape[1])), 2**int(np.log2(original_img.shape[0])))
original_img = cv2.resize(original_img, original_size)

noise_level = 0.3  # Modify by preference

img_noised = add_noise(original_img, noise_level)

cv2.imshow('Original image', original_img)
cv2.imshow('Noised image', np.uint8(img_noised))
cv2.imwrite('./Noised_Lena_BW.jpg', np.uint8(img_noised))
cv2.waitKey(0)
cv2.destroyAllWindows()
