# mataugment/micrographs/transforms.py
import numpy as np
import cv2
from skimage.transform import rotate

class Rotate:
    '''
    Rotate a micrograph by a fixed or random angle.
    If no angle is provided in the input, a random angle between 0–360° is chosen.
    '''
    def __init__(self, angle=None):
        self.angle = angle
    
    def __call__(self, image):
        angle = self.angle if self.angle is not None else np.random.uniform(0, 360)
        rotated = rotate(image, angle, mode='reflect')
        return rotated


class RandomRotate:
    '''
    Rotate a micrograph by a random angle within a given range.
    For Example: RandomRotate(degrees=(-30, 30)) rotates between -30° and +30°.
    '''
    def __init__(self, degrees=(-30, 30)):
        self.degrees = degrees
    
    def __call__(self, image):
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        rotated = rotate(image, angle, mode='reflect')
        return rotated


class Flip:
    '''
    Randomly flip a micrograph horizontally, vertically, or both.
    '''
    def __call__(self, image):
        flip_code = np.random.choice([-1, 0, 1])  # both, vertical, horizontal
        flipped = cv2.flip(image, flip_code)
        return flipped


class RandomCrop:
    '''
    Crop a random region from the image.
    crop_size: (height, width)
    '''
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        h, w = image.shape[:2]
        ch, cw = self.crop_size
        if ch > h or cw > w:
            raise ValueError("Crop size must be smaller than the image size.")
        top = np.random.randint(0, h - ch)
        left = np.random.randint(0, w - cw)
        cropped = image[top:top + ch, left:left + cw]
        return cropped
