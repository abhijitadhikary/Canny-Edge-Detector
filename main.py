import cv2
from canny import run_canny_detector
from utils import load_image, display_image

if __name__ == '__main__':
    filename = 'Leena.png'
    image_input = load_image(filename)
    image_edge = run_canny_detector(image_input)
    display_image('Edge Image (Harris)', image_edge)