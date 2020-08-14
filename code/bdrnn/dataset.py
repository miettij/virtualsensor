import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from os import listdir

class DataSet(Dataset):
    def __init__(self, root_dir, files, normalize=False, seq_len = 1, stride = 1):
        """Takes a list of files as training or test datasets.
        Squeezes the list of files into a 3d array of shape seq_len, examples, (input features + target features + accelerations = 12)"""
        #print(seq_len)
        for file in files:
            if ".DS_Store" in file:
                files.remove(file)
        self.files = files
        self.root_dir = root_dir
        self.normalize = normalize
        self.data_arr = torch.zeros(1,12)
        i=0
        for file in files:
            sequence = extract_seq(self.root_dir, file, self.normalize)
            sequence = split_seq(sequence, seq_len, stride)
            if self.data_arr.size(0) == 1:
                self.data_arr = sequence
            else:
                self.data_arr = torch.cat((self.data_arr, sequence),0)
            #if i == 10:
                #break
            i+=1
        #SAVEPATH = SAVEPATH+'train.csv'
        #arr = self.data_arr.numpy()
        #print("ok", arr)
        #df = pd.DataFrame(arr)
        #df.to_csv(SAVEPATH, sep = ';')
        #a.element_size() * a.nelement()

        #size of array in bytes
        #print("array shape: ",self.data_arr.shape)
        #print("array size (bytes): ",self.data_arr.element_size()*self.data_arr.nelement())

    def __len__(self):
        #return 4 # testdev
        return(self.data_arr.shape[0])

    def __getitem__(self, idx):
        return self.data_arr[idx,:,:]

def split_seq(sequence, seq_len, stride):
    """Gets an input tensor of shape (all_datapoints, data_features)
        Slices the tensor into dimension: (seq_len, F(all_datapoints, seq_len,stride))"""
    seq_len = int(seq_len)
    stride = int(stride)
    start = 0
    end = seq_len
    end_idx = sequence.size(0)-1
    seq = torch.zeros(1,seq_len)

    while end <= end_idx:
        if seq.dim() == 2:
            seq = sequence[start:end, :].unsqueeze(0)
        else:
            temp = sequence[start:end, :].unsqueeze(0)
            seq = torch.cat((seq,temp),0)
        start = start+stride
        end = end+stride
    #print("splitted to seq of shape: ",seq.shape)
    return seq


def extract_seq(root_dir, filename, normalize):
    roll_sim = pd.read_csv(root_dir + filename, header=None, engine='python')
    roll_sim = roll_sim.iloc[1:,[1,2,3,4,5,6,7,8]].copy().to_numpy().T
    #print(roll_sim.shape)
    if normalize:
        #print("normalizing roll_sim arrays")
        roll_sim = normalizematr(roll_sim)

        #print("roll_sim was sliced into n_slices: ", roll_sim.shape)

    roll_sim = torch.from_numpy(roll_sim.T).float()

    return roll_sim

def normalizematr(data):
    #print("normalizematr()", data.shape)

    for i in range(data.shape[0]):
        arr = normalizearr(data[i, :])
        data[i, :] = arr
    return data

def normalizearr(arr1d):
    """Normalizes array to the range [-1,1]"""
    #print("normalizearr()", arr1d.shape)
    arr1d_min = np.min(arr1d)

    arr1d_max = np.max(arr1d)

    arr1d = 2 * (arr1d - arr1d_min) / (arr1d_max - arr1d_min) - 1
    return arr1d
