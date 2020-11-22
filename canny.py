import cv2
from scipy import signal
from utils import *

# run canny detector
def run_canny_detector(im, mode='interpolated', threshold_high=50, threshold_low=20):

    im = im.astype(np.float32) / 255.

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.copy(sobel_x).T

    # apply gaussian filter
    gaussian_filter = gaussian_kernel(11, sigma=11)

    d_gaussian_x = signal.convolve2d(gaussian_filter, sobel_x, boundary='symm', mode='same')
    d_gaussian_y = signal.convolve2d(gaussian_filter, sobel_y, boundary='symm', mode='same')

    # get horizontal and vertical edge images
    dx = signal.convolve2d(im, d_gaussian_x, boundary='symm', mode='same')
    dy = signal.convolve2d(im, d_gaussian_y, boundary='symm', mode='same')

    # combine the two edge images
    magnitude_matrix = np.sqrt(dx ** 2 + dy ** 2)

    # generate the degree matrix by calculating the orientation of the edge
    degree_matrix = np.rad2deg(np.arctan(np.copy(dx), np.copy(dy)))

    # interpolated or quantized
    if mode == 'quantized':
        degree_matrix_quantized = quantize_degrees(degree_matrix)
        suppressed_matrix = perform_non_max_suppression_quantized(magnitude_matrix, degree_matrix_quantized)
    elif mode == 'interpolated':
        suppressed_matrix = perform_non_max_suppression_interpolated(magnitude_matrix, degree_matrix, dx, dy)
    else:
        raise NotImplementedError


    # perform double thresolding
    double_thresholded_matrix = perform_double_threshold(suppressed_matrix, degree_matrix, threshold_high, threshold_low)

    final_edge_image = perform_hysteresis(double_thresholded_matrix)
    final_edge_image = combine_low_high_thresholds(final_edge_image)
    final_edge_image = normalize_image(final_edge_image)

    return final_edge_image

# quantize the edge angles to bins of 0, 45, 90 and 135
def quantize_degrees(degree_matrix):
    degree_matrix = apply_padding(degree_matrix)
    degree_matrix_quantized = np.zeros_like(degree_matrix).astype(np.int32)

    height, width = degree_matrix_quantized.shape

    for row in range(height):
        for column in range(width):
            current_degree = degree_matrix[row, column]
            if (current_degree >= -112.5) and (current_degree < -67.5):
                current_degree_quantized = 90
            elif (current_degree >= -67.5) and (current_degree < -22.5):
                current_degree_quantized = 135
            elif (current_degree >= -22.5) and (current_degree < 22.5):
                current_degree_quantized = 0
            elif (current_degree >= 22.5) and (current_degree < 67.5):
                current_degree_quantized = 45
            elif (current_degree >= 67.5) and (current_degree < 112.5):
                current_degree_quantized = 90

            if not current_degree_quantized is None:
                degree_matrix_quantized[row, column] = current_degree_quantized

    degree_matrix_quantized = remove_padding(degree_matrix_quantized)

    return degree_matrix_quantized

# perform non-maxuimum suppression (quantized form)
def perform_non_max_suppression_quantized(magnitude_matrix, degree_matrix_quantized):
    degree_matrix_quantized = apply_padding(degree_matrix_quantized)
    magnitude_matrix = apply_padding(magnitude_matrix)
    suppressed_matrix = np.zeros_like(magnitude_matrix)

    height, width = suppressed_matrix.shape

    for row in range(1, height - 1):
        for column in range(1, width - 1):
            current_degree = degree_matrix_quantized[row, column]
            current_magnitude = magnitude_matrix[row, column]

            # horizontal edge
            if (current_degree == 90) and ((magnitude_matrix[row, column - 1] > current_magnitude) or (
                    magnitude_matrix[row, column + 1] > current_magnitude)):
                pass
            # vertical edge
            elif (current_degree == 0) and ((magnitude_matrix[row - 1, column] > current_magnitude) or (
                    magnitude_matrix[row + 1, column] > current_magnitude)):
                pass

            # main diagonal edge
            elif (current_degree == 45) and ((magnitude_matrix[row - 1, column - 1] > current_magnitude) or (
                    magnitude_matrix[row + 1, column + 1] > current_magnitude)):
                pass

            # minor diagonal edge
            elif (current_degree == 135) and ((magnitude_matrix[row - 1, column + 1] > current_magnitude) or (
                    magnitude_matrix[row + 1, column - 1] > current_magnitude)):
                pass
            else:
                suppressed_matrix[row, column] = magnitude_matrix[row, column]

    suppressed_matrix = remove_padding(suppressed_matrix)

    return suppressed_matrix

# perform double thresholding, group edges in two categories, strong and medium, discard the weak edge pixels
def perform_double_threshold(suppressed_matrix, degree_matrix, threshold_high, threshold_low):

    # normalize values between [0, 255]
    cv2.normalize(suppressed_matrix, suppressed_matrix, 0, 255, cv2.NORM_MINMAX)

    suppressed_matrix = apply_padding(suppressed_matrix)
    degree_matrix = apply_padding(degree_matrix)

    height, width = suppressed_matrix.shape
    strong_edge_value = 255
    medium_edge_value = 128

    # perform lower threshold and connect
    suppressed_matrix_double_thresholded = np.zeros_like(suppressed_matrix)
    for row in range(1, height-1):
        for column in range(1, width-1):
            current_magnitude = suppressed_matrix[row, column]

            # keep any magnitude greater than the threshold value
            if (current_magnitude > threshold_high):
                suppressed_matrix_double_thresholded[row, column] = strong_edge_value

            # medium edges
            elif (current_magnitude > threshold_low) and (current_magnitude < threshold_high):
                suppressed_matrix_double_thresholded[row, column] = medium_edge_value
            # very weak edges
            else:
                pass

    suppressed_matrix_double_thresholded = remove_padding(suppressed_matrix_double_thresholded)
    return suppressed_matrix_double_thresholded

# check if current pixel has already been traversed
def is_exists(next_coordinates, edge_coordinates_list):
    if next_coordinates in edge_coordinates_list:
        return True
    else:
        return False

# traverse in a clockwise manner to get he next edge pixel starting from the left pixel
def get_next_edge_coordinate(suppressed_dt_edge_image, current_pixel_coordinates, edge_coordinates_list, medium_edge_value):
    row, column = current_pixel_coordinates

    next_coordinates = [row, column - 1]
    if not is_exists(next_coordinates, edge_coordinates_list) and (
            suppressed_dt_edge_image[next_coordinates[0], next_coordinates[1]] >= medium_edge_value):
        return next_coordinates

    next_coordinates = [row - 1, column - 1]
    if not is_exists(next_coordinates, edge_coordinates_list) and (
            suppressed_dt_edge_image[next_coordinates[0], next_coordinates[1]] >= medium_edge_value):
        return next_coordinates

    next_coordinates = [row - 1, column]
    if not is_exists(next_coordinates, edge_coordinates_list) and (
            suppressed_dt_edge_image[next_coordinates[0], next_coordinates[1]] >= medium_edge_value):
        return next_coordinates

    next_coordinates = [row - 1, column + 1]
    if not is_exists(next_coordinates, edge_coordinates_list) and (
            suppressed_dt_edge_image[next_coordinates[0], next_coordinates[1]] >= medium_edge_value):
        return next_coordinates

    next_coordinates = [row, column + 1]
    if not is_exists(next_coordinates, edge_coordinates_list) and (
            suppressed_dt_edge_image[next_coordinates[0], next_coordinates[1]] >= medium_edge_value):
        return next_coordinates

    next_coordinates = [row + 1, column + 1]
    if not is_exists(next_coordinates, edge_coordinates_list) and (
            suppressed_dt_edge_image[next_coordinates[0], next_coordinates[1]] >= medium_edge_value):
        return next_coordinates

    next_coordinates = [row + 1, column]
    if not is_exists(next_coordinates, edge_coordinates_list) and (
            suppressed_dt_edge_image[next_coordinates[0], next_coordinates[1]] >= medium_edge_value):
        return next_coordinates

    next_coordinates = [row + 1, column - 1]
    if not is_exists(next_coordinates, edge_coordinates_list) and (
            suppressed_dt_edge_image[next_coordinates[0], next_coordinates[1]] >= medium_edge_value):
        return next_coordinates

    return None

# check if a strong edge pixel exists in a sequence
def strong_edge_exists(suppressed_dt_edge_image, edge_coordinates_list, strong_edge_value):
    for current_coordinate in edge_coordinates_list:
        # strong edge exists in sequence
        if suppressed_dt_edge_image[current_coordinate[0], current_coordinate[1]] == strong_edge_value:
            return True
    # return false if no strong edge exists in sequence
    return False

# remove a sequence if it does not have a strong edge pixel
def remove_edge_sequence(suppressed_dt_edge_image, edge_coordinates_list):
    for current_coordinate in edge_coordinates_list:
        suppressed_dt_edge_image[current_coordinate[0], current_coordinate[1]] = 0

    return suppressed_dt_edge_image

# joins the low and high thresholds
def combine_low_high_thresholds(image):
    height, width = image.shape

    for row in range(height):
        for column in range(width):
            if image[row, column] == 128:
                image[row, column] = 255
    return image

# performs hysteresis
def perform_hysteresis(suppressed_dt_edge_image):
    suppressed_dt_edge_image = apply_padding(suppressed_dt_edge_image).astype(np.int32)

    height, width = suppressed_dt_edge_image.shape
    strong_edge_value = 255
    medium_edge_value = 128

    # remove medium edges if they are not connected to the strong edges
    for row in range(1, height - 1):
        for column in range(1, width - 1):
            current_magnitude = suppressed_dt_edge_image[row, column]

            if current_magnitude >= medium_edge_value:
                edge_coordinates_list = []
                current_pixel_coordinates = [row, column]
                while True:
                    if not current_pixel_coordinates in edge_coordinates_list:
                        edge_coordinates_list.append(current_pixel_coordinates)

                    # get the next edge pixel
                    current_pixel_coordinates = get_next_edge_coordinate(suppressed_dt_edge_image, current_pixel_coordinates,
                                                                         edge_coordinates_list, medium_edge_value)

                    if current_pixel_coordinates == None:
                        break

                strong_edge_flag = strong_edge_exists(suppressed_dt_edge_image, edge_coordinates_list, strong_edge_value)

                if not strong_edge_flag:
                    suppressed_dt_edge_image = remove_edge_sequence(suppressed_dt_edge_image, edge_coordinates_list)

    suppressed_dt_edge_image = remove_padding(suppressed_dt_edge_image)

    suppressed_dt_edge_image = combine_low_high_thresholds(suppressed_dt_edge_image)

    return suppressed_dt_edge_image

# perform non-maxuimum suppression (interpolated form)
def perform_non_max_suppression_interpolated(magnitude_matrix, degree_matrix, dx, dy):
    degree_matrix = apply_padding(degree_matrix)
    magnitude_matrix = apply_padding(magnitude_matrix)
    suppressed_matrix = np.zeros_like(magnitude_matrix)

    dx = apply_padding(dx)
    dy = apply_padding(dy)

    height, width = suppressed_matrix.shape

    for row in range(1, height - 1):
        for column in range(1, width - 1):
            current_degree = degree_matrix[row, column]
            current_magnitude = magnitude_matrix[row, column]

            if (current_degree >= 0 and current_degree <= 45) or \
                    (current_degree < -135 and current_degree >= -180):

                y_bottom = np.array([magnitude_matrix[row, column + 1],
                                     magnitude_matrix[row + 1, column + 1]])
                y_top = np.array([magnitude_matrix[row, column - 1],
                                  magnitude_matrix[row - 1, column - 1]])

                x_east = np.absolute(dy[row, column] / current_magnitude)

                if (current_magnitude >= ((y_bottom[1] - y_bottom[0]) * x_east + y_bottom[0]) and \
                        current_magnitude >= ((y_top[1] - y_top[0]) * x_east + y_top[0])):
                    suppressed_matrix[row, column] = current_magnitude
                else:
                    suppressed_matrix[row, column] = 0

            elif (current_degree > 45 and current_degree <= 90) or \
                    (current_degree < -90 and current_degree >= -135):

                y_bottom = np.array([magnitude_matrix[row + 1, column],
                                     magnitude_matrix[row + 1, column + 1]])
                y_top = np.array([magnitude_matrix[row - 1, column],
                                  magnitude_matrix[row - 1, column - 1]])

                x_east = np.absolute(dx[row, column] / current_magnitude)

                if (current_magnitude >= ((y_bottom[1] - y_bottom[0]) * x_east + y_bottom[0]) and \
                        current_magnitude >= ((y_top[1] - y_top[0]) * x_east + y_top[0])):
                    suppressed_matrix[row, column] = current_magnitude
                else:
                    suppressed_matrix[row, column] = 0

            elif (current_degree > 90 and current_degree <= 135) or \
                    (current_degree < -45 and current_degree >= -90):

                y_bottom = np.array([magnitude_matrix[row + 1, column],
                                     magnitude_matrix[row + 1, column - 1]])
                y_top = np.array([magnitude_matrix[row - 1, column],
                                  magnitude_matrix[row - 1, column + 1]])

                x_east = np.absolute(dx[row, column] / current_magnitude)

                if (current_magnitude >= ((y_bottom[1] - y_bottom[0]) * x_east + y_bottom[0]) and \
                        current_magnitude >= ((y_top[1] - y_top[0]) * x_east + y_top[0])):
                    suppressed_matrix[row, column] = current_magnitude
                else:
                    suppressed_matrix[row, column] = 0

            elif (current_degree > 135 and current_degree <= 180) or \
                    (current_degree < 0 and current_degree >= -45):

                y_bottom = np.array([magnitude_matrix[row, column - 1],
                                     magnitude_matrix[row + 1, column - 1]])
                y_top = np.array([magnitude_matrix[row, column + 1],
                                  magnitude_matrix[row - 1, column + 1]])

                x_east = np.absolute(dx[row, column] / current_magnitude)

                if (current_magnitude >= ((y_bottom[1] - y_bottom[0]) * x_east + y_bottom[0]) and \
                        current_magnitude >= ((y_top[1] - y_top[0]) * x_east + y_top[0])):
                    suppressed_matrix[row, column] = current_magnitude
                else:
                    suppressed_matrix[row, column] = 0

    suppressed_matrix = remove_padding(suppressed_matrix)

    return suppressed_matrix


