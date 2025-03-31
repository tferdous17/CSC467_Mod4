
import os
import cv2
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from util import generate_gif, renderCube


def split_triptych(trip):
    """
    Split a triptych into thirds
    Input:  trip: a triptych (H x W matrix)
    Output: R, G, B martices
    """
    R, G, B = None, None, None
    # TODO: Split a triptych into thirds and 
    # return three channels as numpy arrays
    return R, G, B


def normalized_cross_correlation(ch1, ch2):
    """
    Calculates similarity between 2 color channels
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
    Output: normalized cross correlation (scalar)
    """



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



def pyramid_align():
    # TODO: Reuse the functions from task 2 to perform the 
    # image pyramid alignment iteratively or recursively



def main():
    # TODO: Solution for Q2
    # Task 1: Generate a colour image by splitting 
    # the triptych image and save it 

    # Task 2: Remove misalignment in the colour channels 
    # by calculating best offset
    
    # Task 3: Pyramid alignment



if __name__ == "__main__":
    main()
