import cv2
import numpy as np
from wiener_filter import adaptive_wiener_filter
from build_laplacian_pyramid import build_laplacian_pyramid
from add_noise import add_noise


def pseudo_cross_bilateral_filter(img1, img2, sigma_spatial, sigma_range):
    rows, cols = img1.shape[:2]
    filtered_img = np.zeros_like(img1, dtype=np.float32)

    for y in range(rows):
        for x in range(cols):
            # Spatial weight
            spatial_weight = np.exp(-((y - rows // 2) ** 2 + (x - cols // 2) ** 2) / (2 * sigma_spatial ** 2))

            # Intensity difference
            intensity_diff = np.abs(img1[y, x] - img2[y, x])

            # Range weight
            range_weight = np.exp(-np.clip(intensity_diff / (2 * sigma_range ** 2), -700, 700))

            # Combined weight (normalize to ensure weights sum up to 1.0)
            total_weight = spatial_weight + range_weight
            combined_weight = range_weight / total_weight

            # Apply the weight to each channel of the second image
            filtered_img[y, x] = combined_weight * img2[y, x] + (1 - combined_weight) * img1[y, x]

    return np.uint8(filtered_img)


# Example for all levels saved in a dir
no_levels = 4
sigma_spatial = 2.0
sigma_range = 20.0
original_img = cv2.imread('./Lena_BW.jpg', cv2.IMREAD_GRAYSCALE)
noised_img = add_noise(original_img)
original_img_wiener = adaptive_wiener_filter(noised_img)
img1 = build_laplacian_pyramid(noised_img, no_levels)
img2 = build_laplacian_pyramid(original_img_wiener, no_levels)
for i in range(no_levels + 1):
    filtered_image = pseudo_cross_bilateral_filter(img1[i], img2[i], sigma_spatial, sigma_range)

    # Display and save the filtered result
    # cv2.imshow(f'Pseudo Cross for level {i}', filtered_image)
    cv2.imwrite(f'./bilateral_filter/level_{i}.png', filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
