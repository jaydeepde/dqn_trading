import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
#import cPickle
## import cv2

from joblib import Parallel, delayed
import multiprocessing


# convert a time series data into Gramian Angular Field Image
# and resize it to 50 X 50
##def to_gaf(state):
##    # convert state to GAF
##    g = state
##    g1 = np.outer(g, np.ones((state.shape[0],)))
##    g2 = np.outer(np.ones((state.shape[0],)), g)
##    gaf = cv2.resize(np.cos(g1+g2),dsize=(50,50), interpolation = cv2.INTER_CUBIC)
##    return gaf


def to_1d(state):
    # convert state to 1d data
    return state
# function for reading the all the timeseries
# data parallaly and converting them into GAF image 

# input = file_to_read (column 0 = epoch time, column 1+ = data)
def timeseries_into_images(file_to_read,no_of_points_per_hour):
    
    window_length = 12 * no_of_points_per_hour 
    # read as pandas dataframe
    data = pd.read_csv(file_to_read)

    # convert the data in data time series
    timeindex = pd.to_datetime(data.ix[::, 0].values, unit='s')
    
    
    # data for saving all the images
    all_data = []
    # all_data_1D = []
    
    for ch_num in range(data.shape[1]-1):
        # extract the product of interest
        data_time_series = pd.Series(data.ix[::, ch_num + 1].values, index=timeindex)
    
        # resample in N-Min by last value
        str1 = str(np.absolute(60/no_of_points_per_hour)) + 'T'
        data_resample = data_time_series.resample(str1).last()
    
        # remove the nans
        data_resample = data_resample.dropna()
    
        # covert the data in numpy format
        data_resample_numpy = np.float32(data_resample.ix[:].values)
    
        data_size = data_resample.size
        #data_size = 1000 + window_length
    
        # parallaly process the data
        num_cores = multiprocessing.cpu_count()
        return_data = Parallel(n_jobs=num_cores,verbose=2)(delayed(to_gaf)\
        (data_resample_numpy[idx:idx + window_length]) for idx in range(0, data_size - window_length))

    all_data.append(return_data)
    #all_data_1D.append(oneD_data)
    
    
    return data_resample_numpy, all_data
    
def timeseries_into_1d(file_to_read,no_of_points_per_hour):
    
    window_length = 12 * no_of_points_per_hour 
    # read as pandas dataframe
    data = pd.read_csv(file_to_read)

    # convert the data in data time series
    timeindex = pd.to_datetime(data.ix[::, 0].values, unit='s')
    
    
    # data for saving all the images
    all_data = []
    # all_data_1D = []
    
    for ch_num in range(data.shape[1]-1):
        # extract the product of interest
        data_time_series = pd.Series(data.ix[::, ch_num + 1].values, index=timeindex)
    
        # resample in N-Min by last value
        str1 = str(np.absolute(60/no_of_points_per_hour)) + 'T'
        data_resample = data_time_series.resample(str1).last()
    
        # remove the nans
        data_resample = data_resample.dropna()
    
        # covert the data in numpy format
        data_resample_numpy = np.float32(data_resample.ix[:].values)
    
        data_size = data_resample.size
        #data_size = 1000 + window_length
    
        # parallaly process the data
        num_cores = multiprocessing.cpu_count()
        return_data = Parallel(n_jobs=num_cores,verbose=2)(delayed(to_1d)\
        (data_resample_numpy[idx:idx + window_length]) for idx in range(0, data_size - window_length))

    all_data.append(return_data)
    #all_data_1D.append(oneD_data)
    
    
    return data_resample_numpy, all_data
