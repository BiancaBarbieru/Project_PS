import cv2
from build_laplacian_pyramid import build_laplacian_pyramid


def denoising_per_levels(laplacian_level):
    filtered_level = cv2.medianBlur(laplacian_level, 5)

    return filtered_level


original_img = cv2.imread('./Noised_Lena_BW.jpg', cv2.IMREAD_GRAYSCALE)

no_levels = 4

laplacian_pyramid = build_laplacian_pyramid(original_img, no_levels)

filtered_pyramid = [denoising_per_levels(level) for level in laplacian_pyramid]

for i, level in enumerate(filtered_pyramid):
    # cv2.imshow(f'Filtered Level {i}', level)
    cv2.imwrite(f'./filtered_levels/level_{i}.png', level)

cv2.waitKey(0)
cv2.destroyAllWindows()
