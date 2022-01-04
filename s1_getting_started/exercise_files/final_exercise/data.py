import torch
from torchvision import datasets, transforms
import numpy as np
from numpy import load, concatenate

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self,file_names,path='C:/Users/Bruger/Documents/MLops/dtu_mlops-main/data/corruptmnist/',verbose=False):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
        for i in range(len(file_names)):
            if (i==0):
                data = load(path+file_names[0])
                if verbose:
                    for f in data.files:
                        print(f)
                        print(type(data[f]))
                self.images = data['images']
                self.labels = data['labels']
            else:
                data = load(path+file_names[i])
                if verbose:
                    for f in data.files:
                        print(f)
                        print(type(data[f]))
                self.images = concatenate([self.images,data['images']])
                self.labels = concatenate([self.labels,data['labels']])
        self.images = self.images.astype(np.float)
        self.labels = self.labels.astype(np.float)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def mnist():

    # Download and load the training data
    train = CustomDataSet(['train_0.npz','train_1.npz','train_2.npz','train_3.npz','train_4.npz'])
    trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    test = CustomDataSet(['test.npz'])
    testloader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)
    return trainloader, testloader

# if __name__ == '__main__':
#     train, test = mnist()
#     for ite,batch in enumerate(train):
#         images, labels = batch
#         print(labels)
