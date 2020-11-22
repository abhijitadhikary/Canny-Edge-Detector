import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# returns dimentions of an image
def get_dimentions(input_image):
    # for rgb images
    if len(input_image.shape) == 3:
        input_height, input_width, input_channel = input_image.shape

    # for grayscale images
    elif len(input_image.shape) == 2:
        input_height, input_width = input_image.shape
        input_channel = 1

    return input_height, input_width, input_channel


# applies padding of given amount to an image and returns the padded image
def apply_padding(input_image, pad_height=1, pad_width=1):
    input_height, input_width, input_channel = get_dimentions(input_image)

    output_height = np.ceil(input_height + pad_height * 2).astype(int)
    output_width = np.ceil(input_width + pad_width * 2).astype(int)

    # for gray images
    if input_channel == 1:
        output_image = np.zeros((output_height, output_width))
    # for rgb images
    elif input_channel == 3:
        output_image = np.zeros((output_height, output_width, input_channel))

    output_image[pad_height:input_height + pad_height, pad_width:input_width + pad_width] = input_image
    return output_image

# removes padding from an image
def remove_padding(input_image, pad_height=1, pad_width=1):
    input_height, input_width, input_channel = get_dimentions(input_image)
    output_image = input_image[pad_height:input_height - 1, pad_width:input_width - 1]

    return output_image

# generate a gaussian kernel
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return kernel

# converts pixel values between 0 and 255
def normalize_image(image):
    image = image / np.max(image)
    image = image * 255.
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image

# load an image and convert to grayscale
def load_image(filename):
    img = cv2.imread(os.path.join(filename))
    # img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# display image using opencv
def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# display an image using matplotlib
def imshow(im, title='', figsize=(10, 10), cmap='gray'):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(im, cmap=cmap)
    plt.plot()
    plt.show()