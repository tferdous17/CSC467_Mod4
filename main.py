
import os
import cv2
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
# from util import generate_gif, renderCube


def split_triptych(trip):
    """
    Split a triptych into thirds
    Input:  trip: a triptych (H x W matrix)
    Output: R, G, B martices
    """
    # TODO: Split a triptych into thirds and
    # return three channels as numpy arrays

    height, width = trip.shape

    # calculate the height of each image
    third_height = height // 3

    # top third, B
    B = trip[0:third_height, :]
    # middle third, G
    G = trip[third_height:2 * third_height, :]
    # bottom third, R
    R = trip[2 * third_height:3 * third_height, :]

    return R, G, B


def normalized_cross_correlation(ch1, ch2):
    """
    Calculates similarity between 2 color channels
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
    Output: normalized cross correlation (scalar)
    """
    ch1_flattened = ch1.flatten()
    ch2_flattened = ch2.flatten()

    ch1_normalized = ch1_flattened / np.linalg.norm(ch1_flattened)
    ch2_normalized = ch2_flattened / np.linalg.norm(ch2_flattened)

    normalized_cc = np.dot(ch1_normalized, ch2_normalized)
    return normalized_cc



def best_offset(ch1, ch2, metric, Xrange=np.arange(-10, 10), 
                Yrange=np.arange(-10, 10)):
    """
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
            metric: similarity measure between two channels
            Xrange: range to search for optimal offset in vertical direction
            Yrange: range to search for optimal offset in horizontal direction
    Output: optimal offset for X axis and optimal offset for Y axis

    Note: Searching in Xrange would mean moving in the vertical 
    axis of the image/matrix, Yrange is the horizontal axis 
    """
    # TODO: Use metric to align ch2 to ch1 and return optimal offsets

    best_score = float('-inf')
    optimal_x = 0
    optimal_y = 0

    for x in Xrange:
        for y in Yrange:
            ch2_shifted = np.roll(ch2, x, axis=0)
            ch2_shifted = np.roll(ch2_shifted, y, axis=1)

            sim_score = metric(ch1, ch2_shifted)
            if sim_score > best_score:
                best_score = sim_score
                optimal_x = x
                optimal_y = y

    return optimal_x, optimal_y


def align_and_combine(R, G, B, metric):
    """
    Input:  R: red channel
            G: green channel
            B: blue channel
            metric: similarity measure between two channels
    Output: aligned RGB image 
    """
    # TODO: Use metric to align the three channels 
    # Hint: Use one channel as the anchor to align other two

    # will use R channel as anchor
    # align G to R
    gx, gy = best_offset(R, G, metric)
    G_aligned = np.roll(G, gx, axis=0)
    G_aligned = np.roll(G_aligned, gy, axis=1)

    # align B to R
    bx, by = best_offset(R, B, metric)
    B_aligned = np.roll(B, bx, axis=0)
    B_aligned = np.roll(B_aligned, by, axis=1)

    # combine aligned channels
    aligned_image = np.stack([R, G_aligned, B_aligned], axis=2)

    return aligned_image, (gx, gy), (bx, by)



def pyramid_align():
    # TODO: Reuse the functions from task 2 to perform the
    # image pyramid alignment iteratively or recursively
    pass




def main():
    # TODO: Solution for Q2

    # Task 1: Generate a colour image by splitting 
    # the triptych image and save it
    trip = plt.imread('01112v.jpg')
    R, G, B = split_triptych(trip)
    colored_img = np.stack([R, G, B], axis=2)
    plt.imsave('01112v_COLORIZED.jpg', colored_img)

    # Task 2: Remove misalignment in the colour channels 
    # by calculating best offset
    aligned_img, g_offset, b_offset = align_and_combine(R, G, B, normalized_cross_correlation)
    plt.imsave('01112v_aligned.jpg', aligned_img)
    
    # Task 3: Pyramid alignment



if __name__ == "__main__":
    main()
