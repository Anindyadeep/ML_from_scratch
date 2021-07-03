import numpy as np 
from ml_regression_utils import universal_reshape

def OneHotEncode(y):
    """
    This function helps to change the labels to its corersponding 
    One Hot Encodings
    """
    _, y = universal_reshape(y,y)

    shape=(y.shape[1], y.max()+1)
    OHE = np.zeros(shape)
    rows = np.arange(y.shape[1])
    OHE[rows, y]=1
    
    return OHE

