import cv2 
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    image = cv2.resize(image, (500,800))
    cv2.imshow('Original', image)
    cv2.waitKey(0)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray_image)
    cv2.waitKey(0)

    m, n = gray_image.shape
    min_dim = min(m, n)
    
    return gray_image, min_dim

def low_rank_approximation(A, k):
    U, S, V_t = np.linalg.svd(A)
    approx = U[:, :k] @ np.diag(S[:k]) @ V_t[:k, :]
    return approx

def plot_approximations(image, min_dim):
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    i = 1
    k = 2
    while i < min_dim:
        approx_image = low_rank_approximation(image, i)
        plt.subplot(3, 4, k)
        plt.imshow(approx_image, cmap='gray')
        plt.title(f'n = {i}')
        i *= 2
        k += 1

    approx_image = low_rank_approximation(image, min_dim)
    plt.subplot(3, 4, k)
    plt.imshow(approx_image, cmap='gray')
    plt.title(f'n = {min_dim}')

    plt.tight_layout()
    plt.show()


image = cv2.imread('image.jpg')
gray_image, min_dim = preprocess_image(image)

plot_approximations(gray_image, min_dim)

print("lowest k = 32")


# d = min_dim // 10
# i = 1
# while i < min_dim:
#     approx = low_rank_approximation(gray_image, i)
#     print("Rank: ", i)
#     # Normalize values for display
#     approx_display = cv2.normalize(approx, None, 0, 255, cv2.NORM_MINMAX)
    
#     cv2.imshow('Approximation', approx_display.astype(np.uint8))
#     cv2.waitKey(0)
#     i *= 2


# Window shown waits for any key pressing event
cv2.destroyAllWindows()



