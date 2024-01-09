import cv2
import numpy as np


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

    return filtered_img.astype(np.uint8)


# Example for all levels saved in a dir
no_levels = 4
sigma_spatial = 2.0
sigma_range = 20.0
for i in range(no_levels + 1):
    img1 = cv2.imread(f'./laplacian_levels_noised/level_{i}.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f'./laplacian_levels_wiener/level_{i}.png', cv2.IMREAD_GRAYSCALE)

    filtered_image = pseudo_cross_bilateral_filter(img1, img2, sigma_spatial, sigma_range)

    # Display and save the filtered result
    # cv2.imshow(f'Pseudo Cross for level {i}', filtered_image)
    cv2.imwrite(f'./bilateral_filter/level_{i}.png', filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
