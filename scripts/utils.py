import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


############################ STATS input data ################################################


def return_nan_percentage(input_data):
    """
    prints percentage of nan values in max. 3D sized array

    Parameters
    ----------
    input_array : array
       max 3D array

    Returns
    -------
    None

    """
    total_size = input_data.size
    nan_sum = np.isnan(input_data).sum()
    perc = float(nan_sum / total_size)
    print("percentage of nan values inside dataset is: %.2f" % float(perc) + " %")


# #4D example:
# for i in Training_data:
#     return_nan_percentage(i)


def describe_with_stats(input_data):
    flat_array = input_data.flatten()
    # 'omit'performs the calculations ignoring nan values
    nobs, minmax, mean, variance, skewness, kurtosis = stats.describe(
        flat_array, nan_policy="omit"
    )
    print("Number of observations: " + str(nobs))
    print("min: " + str(minmax[0]))
    print("max: " + str(minmax[1]))
    print("the mean is: " + str(mean))
    print("the variance is: " + str(variance))
    print("Skewness is: " + str(skewness))
    print("Kurtosis: " + str(kurtosis))
    print("---")


# for i in Training_data_germany:
#     describe_with_stats(i)

############################ Derive Labels ###############################################


def mask_nan_values(input_array):
    array_with_masked_nans = input_array.fillna(value=10000.00)
    return array_with_masked_nans


# back to xarray with:
# label_xarray = xr.DataArray(output_3D_array, dims=['time', 'latitude', 'longitude']  )


# to turn list output into a 3D array use:
def list_to_array(output_list):
    output_3D_array = np.stack(output_list, axis=0)
    return output_3D_array


# TODO: returns list of 2D arrays now, try to return 3D x array to save as net cdf -SEE BELOW
# TODO: write test


# #Example:
# #create data subset of 10 of a data xarray
# data_10 = data[0:10] #first 10 items to test
# print(data.shape)

# #call function with a threshod of 10
# output_array = binary_image_classification(data_10, T=0.5)

# #show one image of the masked output images
# plt.imshow(output_array[0], origin = 'lower')
# #might need to change 'lower' to 'upper"


# TODO:
def list_of_2D_xarray_to_netcdf():
    x = 0
    netcdf = x
    return netcdf


def save_plots_from_3Darray(
    input_array, OUTPUT_PATH, title="drought mask figure Nr:", show_plots=True
):
    """
    saves pngs and/or prints images from 3Darrays as png files

    Parameters
    ----------
    input_xarray : array
        3-D input array in the format [num_samples, height, width]
    title: str
        title of the plots, number will be added according to iteration index
    show_plots: boolean
        determines if plots will be displayed as output or not


    Returns
    -------
    None


    """

    for k in range(len(input_array[0])):
        fig = input_array[k].plot()
        plt.title(title + str(k))
        plt.axis("equal")
        plt.title("drought mask for SMI, month " + str(k))
        if show_plots:
            plt.show()
        fig.figure.savefig(OUTPUT_PATH + title + str(k) + ".png", dpi=100)
        print(OUTPUT_PATH + "drought_mask_" + str(k) + ".png")


############################ class imbalance ######################################


# option 1, faster, combine these 2 fcts (recommended):


def hide_random_values(input_value, T=0.68):
    if input_value == 0:
        if np.random.rand(1) > T:
            return -1
    return input_value


# print(hide_random_values(0))


def reduce_class_size(input_array):
    output_array = np.copy(input_array)
    for t in range(0, 472):
        for xy in range(0, 7171):
            output_array[t, xy] = hide_random_values(output_array[t, xy])

    return output_array


# option 2, combine these 2 fcts:


def get_indices(dataset, value=0):
    """dataset = str(), 2D-array
    value = int(), value to print the indices for"""
    result = np.where(dataset == value)
    print("Tuple of arrays returned : ", result)
    # zip the 2 arrays (array 1: rows, array 2: columns) to get the exact coordinates
    listOfCoordinates = list(zip(result[0], result[1]))
    # iterate over the list of coordinates
    # for cord in listOfCoordinates:
    #     print(cord)
    print(len(listOfCoordinates))
    return listOfCoordinates


def reduce_class_size(input_array, indices_list, T=0.78, value=int(-1)):
    """set entries in array to value=x, randomly and within set percentage of array

    list = list, list of indices (2D)
    T = int() , percentage to be modified

    returns:

    """
    output_array = np.copy(input_array)

    # determine the percentage of the array that will be modified
    len_modifier = int(len(indices_list) * T)

    # select percentage T randomly from the list
    random_coords = random.sample(listOfCoordinates, len_modifier)
    # print(random_coords[:10])
    # set selected entries to value
    print("selected indices will be set to " + str(value))
    for i in random_coords:
        # print(labels_reshaped[i])
        output_array[i] == value
    return output_array
