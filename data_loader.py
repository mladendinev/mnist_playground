from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
# training_data = datasets.MNIST(root='./data', train = True, download=True, transform=ToTensor())
# testing_data = datasets.MNIST(root ='./data', train = False, download=True, transform=ToTensor())

from dataset import MNISTTestDataset, MNISTTrainDataset
training_data = MNISTTrainDataset()
testing_data = MNISTTestDataset()
training_loader = DataLoader(training_data, batch_size=10, shuffle=True)
testing_loader = DataLoader(testing_data, shuffle=False)



