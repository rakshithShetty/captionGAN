import numpy as np

class numpy_list_data:
  def __init__(self, path):
    self.featlist = open(path,'r').read().splitlines()

  def get_float_list(self, iL):
    assert(len(iL) == 1)
    y = np.load(self.featlist[iL[0]])['x']
    return y.reshape((y.shape[0],-1)).T



