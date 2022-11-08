import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count
from jobdispatcher.index_based_loops import pFor, Static

from copy import deepcopy


def _convert_single(
    n: int, X: np.ndarray, resolution: int, padding_ratio: float
) -> np.ndarray:
    """
    Simple function based around matplotlib capable of converting a given feature vector from
    the dataset into a black-and-white image. The function has been defined to be used together
    with the index based parallel for function implemented in jobdispatcher.

    Arguments
    ---------
    n: int
        The integer index of the feature vector in the dataset X.
    X: np.ndarray
        The whole dataset from which the image must be generated. The function expects an array
        containing a set of features vector of shape `2*N` in which the first row encodes the
        real part of the impedance while the second its imaginary part.
    resolution: int
        The number of pixel to be used in the conversion.
    padding_ration: float
        The percentual of the image dedicated to padding. (Required to avoid the cut of pixels
        located on the edge of the image)
    
    Returns
    -------
    np.ndarray
        The array of np.bool_ encoding the black and white image
    """

    # Extract the desired feature vector from the dataset and read the real and imaginary part
    x = X[n]
    re, im = x[0, :], -x[1, :]

    # Compute a the width of the padding_ratio region to apply around the picture
    delta = [padding_ratio * (max(re) - min(re)), padding_ratio * (max(im) - min(im))]
    limits = [
        [min(re) - delta[0], max(re) + delta[0]],
        [min(im) - delta[1], max(im) + delta[1]],
    ]

    # Create a figure without a frame and set the desired resolution
    fig = plt.figure(frameon=False, dpi=resolution)
    fig.set_size_inches(1.0, 1.0)

    # Create an axes object, plot the data and turn off all the axis, frame, ticks ...
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.plot(re, im, linewidth=0.01, c="black")
    ax.set_axis_off()
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    fig.add_axes(ax)

    # Draw the figure in a canvas
    fig.canvas.draw()

    # Extract the data from the canvas as integer rgb values from 0 to 255 and reshape
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Extract the black and white image using the built in numpy broadcast property
    bwdata = np.array(1 - (np.mean(data, axis=2) / 255.0), dtype=np.bool_)

    # Close the figure and free the memory
    plt.close()
    del fig
    del data

    return bwdata


def to_image(
    dataset_X: np.ndarray,
    resolution: int,
    is_polar: bool = False,
    padding_ratio: float = 0.05,
    ncores: int = cpu_count(),
) -> np.ndarray:
    """
    Simple function capable of converting an impedance value based dataset into a dataset
    composed by images.
    
    Arguments
    ---------
    dataset_X: np.ndarray
        The whole impedance dataset from which the image dataset must be generated. 
    resolution: int
        The number of pixel to be used in the conversion.
    padding_ration: float
        The percentual of the image dedicated to padding. (Required to avoid the cut of pixels
        located on the edge of the image)
    ncores: int
        The nomber of CPU to be used for the parallel computation. (default: multiprocessing.cpu_count)
    
    Returns
    -------
    np.ndarray
        The array of dataset images.
    """

    # Create a local copy of the provided dataset
    X = deepcopy(dataset_X)

    # Evaluate the original shape of the input and, if necessary, change the shape to a 2D vector
    shape = X.shape
    if len(shape[1::]) == 1:
        X = X.reshape([shape[0], 2, int(shape[1] / 2)])

    # If the representation is polar change it to cartesian
    if is_polar:

        buffer = []
        for x in X:
            Z = x[0, :] * np.exp(1j * x[1, :])
            b = np.concatenate((Z.real, Z.imag), axis=0)
            b = b.reshape(x.shape)
            buffer.append(b)

        X = np.array(buffer)

    output = pFor(Static(ncores))(
        _convert_single, 0, len(X), args=[X, resolution, padding_ratio]
    )

    return np.array(output)
