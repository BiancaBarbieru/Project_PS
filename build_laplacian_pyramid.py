import cv2


def build_laplacian_pyramid(img, levels):
    laplacian_pyramid = [img]

    # Iterate through the levels to create the pyramid
    for level in range(levels):
        # Generate a downsampled version of the image
        img_pyramid = cv2.pyrDown(img)

        # Reconstruct the image by upsampling the downsampled version
        reconstructed_img = cv2.pyrUp(img_pyramid, dstsize=(img.shape[1], img.shape[0]))

        # Calculate the Laplacian level by subtracting the reconstructed image from the original
        laplacian_level = cv2.subtract(img, reconstructed_img)

        # Append the Laplacian level to the pyramid
        laplacian_pyramid.append(laplacian_level)

        # Update the image for the next iteration
        img = img_pyramid

    return laplacian_pyramid


# Read the original noised image
original_img = cv2.imread('./Noised_Lena_BW.jpg', cv2.IMREAD_GRAYSCALE)

# Specify the number of levels in the Laplacian pyramid
no_levels = 4

# Build the Laplacian pyramid
pyramid = build_laplacian_pyramid(original_img, no_levels)

# Save each level of the Laplacian pyramid as an individual image
for i, level in enumerate(pyramid):
    cv2.imwrite(f'./laplacian_levels_noised/level_{i}.png', level)
cv2.waitKey(0)


# Same for decomposing the image filtered with wiener just to save the steps
original_img = cv2.imread('./Wiener_Filter_Over_Noised_Lena.jpg', cv2.IMREAD_GRAYSCALE)
no_levels = 4
pyramid = build_laplacian_pyramid(original_img, no_levels)
for i, level in enumerate(pyramid):
    cv2.imwrite(f'./laplacian_levels_wiener/level_{i}.png', level)
cv2.waitKey(0)


# Close all windows
cv2.destroyAllWindows()
