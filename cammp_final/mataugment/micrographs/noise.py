# mataugment/micrographs/noise.py
import numpy as np
import cv2

class AddGaussianNoise:
# Add Gaussian noise to simulate real-world imperfectinos 
    def __init__(self, mean=0, std=0.02):
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        noise = np.random.normal(self.mean, self.std, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1)

class AddExperimentalNoise:
  #Blending experimental noise with the clean image
    def __init__(self, noise_image_path, alpha=0.3):
        self.noise_image = cv2.imread(noise_image_path, cv2.IMREAD_GRAYSCALE) / 255.0
        self.alpha = alpha

    def __call__(self, image):
        if image.shape != self.noise_image.shape:
            noise_resized = cv2.resize(self.noise_image, (image.shape[1], image.shape[0]))
        else:
            noise_resized = self.noise_image
        return np.clip((1 - self.alpha) * image + self.alpha * noise_resized, 0, 1)
