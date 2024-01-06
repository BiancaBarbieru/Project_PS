import cv2


def build_laplacian_pyramid(img, levels):
    laplacian_pyramid = [img]

    for level in range(levels):
        img_pyramid = cv2.pyrDown(img)
        reconstructed_img = cv2.pyrUp(img_pyramid, dstsize=(img.shape[1], img.shape[0]))

        laplacian_level = cv2.subtract(img, reconstructed_img)
        laplacian_pyramid.append(laplacian_level)

        img = img_pyramid

    return laplacian_pyramid


original_img = cv2.imread('./Noised_Lena_BW.jpg', cv2.IMREAD_GRAYSCALE)

no_levels = 4

pyramid = build_laplacian_pyramid(original_img, no_levels)

for i, level in enumerate(pyramid):
    # cv2.imshow(f'Level {i}', level)
    cv2.imwrite(f'./laplacian_levels/level_{i}.png',  level)
cv2.waitKey(0)
cv2.destroyAllWindows()
