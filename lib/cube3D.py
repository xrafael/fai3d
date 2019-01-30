
import numpy as np

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def clip(data, verbose=0):
    if verbose > 0:
        print("\nData before clip:", data.shape, np.max(data), np.min(data))

    # make image between 0 .. 1 corresponding to HU values -1000 to 400
    data[data < MIN_BOUND] = MIN_BOUND
    data[data > MAX_BOUND] = MAX_BOUND

    if verbose > 0:
        print("Data after clip:", data.shape, np.max(data), np.min(data))

    return data

def normalize(data, verbose=0):
    if verbose > 0:
        print("\nData before normalize:", data.shape, np.max(data), np.min(data))

    # make image between 0 .. 1 corresponding to HU values -1000 to 400
    data = (data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    if verbose > 0:
        print("Data after normalize:", data.shape, np.max(data), np.min(data))

    return data

def open_cube(loc):
    cube = np.float32(np.load(loc)[0])
    cube = clip(cube)
    cube = normalize(cube)
    return cube
