# mataugment/micrographs/utils.py
import cv2
import matplotlib.pyplot as plt

def load_micrograph(path):
    #Load image as grayscale normalized array.
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype('float32') / 255.0

def save_micrograph(image, path):
    #Save image to path (0–255).
    cv2.imwrite(path, (image * 255).astype('uint8'))

def visualize(image, title="Micrograph"):
    #Quick visualization of the image.
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
