import torch
import pandas as pd
import numpy as np

from torch.utils import data

class FaceDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, folder_name, training = True, transform = None):
        'Initialization'
        self.transform = transform
        
        if training:
            face_data = np.array(pd.read_csv(folder_name+"/train_face.csv", index_col=None, header=None))
            nface_data = np.array(pd.read_csv(folder_name+"/train_nface.csv", index_col=None, header=None))
        else:
            face_data = np.array(pd.read_csv(folder_name+"/test_face.csv", index_col=None, header=None))
            nface_data = np.array(pd.read_csv(folder_name+"/test_nface.csv", index_col=None, header=None))
        
        face_label = np.ones(len(face_data), dtype=int)
        nface_label = np.zeros(len(nface_data), dtype=int)
        
        self.data = np.concatenate([face_data, nface_data], axis=0)
        self.label = np.concatenate([face_label, nface_label], axis=0)
        
        #divide by 255
        self.data = self.data / 255.0
        
        #reshape
        self.data = self.data.reshape(-1, 16, 16).astype(np.float32)
        
        shuffle_index = np.arange(len(self.data))
        np.random.shuffle(shuffle_index)
        
        self.data = self.data[shuffle_index]
        self.label = self.label[shuffle_index]
        
        return

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.data[index]
        y = self.label[index]
        
        if self.transform is not None:
            X = self.transform(X)

        return X, y
    
    