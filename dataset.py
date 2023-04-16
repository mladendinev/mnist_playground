from abc import ABC
import pickle
import gzip 
import numpy as np
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        f = gzip.open('./data/mnist.pkl.gz', 'rb')
        self.training_data, self.validation_data, self.test_data = pickle.load(f, encoding="latin1")
        f.close()
    
    def vectorized_result(self, j):
        """Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network."""
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
    
    
class MNISTTrainDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.training_inputs = [np.reshape(x, (784, 1)) for x in self.training_data[0]]
        self.training_labels = [self.vectorized_result(y) for y in self.training_data[1]]
        
    def __len__(self):
        return len(self.training_data[0])
    
    def __getitem__(self, idx):   
        return (self.training_inputs[idx], self.training_labels[idx])
    

class MNISTTestDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.test_inputs = [np.reshape(x, (784, 1)) for x in self.test_data[0]]
        self.test_labels = self.test_data[1]
        
    def __len__(self):
        return len(self.test_data[0])
    
    def __getitem__(self, idx):
        return (self.test_inputs[idx], self.test_labels[idx])
