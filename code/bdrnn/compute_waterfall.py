import numpy as np
from utils import read_names, make_dict, ascendingorder_wf, read_params, find_closest, get_freq_domain, get_freq_domain2, scale_seqs
from dataset import DataSet
import torch
from model import LSTM_layers
from torch.utils.data import DataLoader
import torch.nn.functional as F

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
seq_len = 10000
max_stride = 1000
limit = 4000

def fft(sequence1,y1, velocity, seq_len, filename,minmaxdatapath):
    sequence1 = sequence1.detach().squeeze(1).cpu().numpy() #first of the minibatch
    y1 = y1.detach().squeeze(1).cpu().numpy()

    sequence1,y1 = scale_seqs(sequence1,y1,filename,minmaxdatapath)

    sequence1,y1 = sequence1[0],y1[0]
    #frq, Y = get_freq_domain2(sequence1[:,0],velocity) #Horizontal
    #frq_true, Y_true = get_freq_domain2(y1[:,0],velocity)
    #frq, Y = get_freq_domain2(sequence1[:,1],velocity) #Vertical
    #frq_true, Y_true = get_freq_domain2(y1[:,1],velocity)
    frq, Y = get_freq_domain(sequence1[:,0])
    frq_true, Y_true = get_freq_domain(y1[:,0])
    return frq, np.abs(Y), frq_true, np.abs(Y_true)

def waterfall():
    filepath = '../../original/processed_data/'
    minmaxdatapath = '../../original/minmaxdata/'
    filenamepath = '../../scratch/bd_lstm/filenames/testfiles.txt'
    weightpath = '../../scratch/bd_lstm/trainstats/weights_middle.pth'
    parampath = './conf_model.cfg'

    filenames = read_names(filenamepath)

    filenamedict = make_dict(filenames)

    vels = ascendingorder_wf(filenames)
    num_files = len(vels)

    params = read_params(parampath)
    model = LSTM_layers(
        input_size=int(params['input_size']),
        hidden_size=int(params['hidden_size']),
        num_layers=int(params['n_layers']),
        dropout=float(params['dropout']),
        output_size=int(params['output_size']),
        batch_first = True,
        bidirectional = True)

    model.load_state_dict(torch.load(weightpath, map_location = 'cpu'))
    model.to(device)
    model.eval()
    arr = None
    hack_idx = 0
    for velocity in vels:
        filename, velocity = find_closest(filenamedict, velocity)

        files = [filename]
        dataset = DataSet(root_dir = filepath, files = files, normalize = False, seq_len = seq_len, stride = max_stride)
        loader = DataLoader(dataset, batch_size = int(params['batch_size']), shuffle = True)
        for idx, sample in enumerate(loader):
            y = sample[:,:,:2].clone().detach().requires_grad_(True).to(device)
            x = sample[:,:,2:].clone().detach().requires_grad_(True).to(device)
            h0 = model.init_hidden(int(params['batch_size']),None).to(device)
            c0 = model.init_cell(int(params['batch_size'])).to(device)

            #compute
            output = model.forward(x,(h0,c0))
            frq_pred, Y_pred, frq_true, Y_true = fft(output, y, velocity, seq_len,filename,minmaxdatapath)
            vel_pred = np.ones(len(frq_pred))*velocity
            break
        if hack_idx == 0:
            arr_pred = np.vstack((vel_pred,frq_pred,Y_pred))
            arr_true = np.vstack((vel_pred,frq_true, Y_true))
        else:
            arr2_pred = np.vstack((vel_pred,frq_pred,Y_pred))
            arr2_true = np.vstack((vel_pred,frq_true,Y_true))
            arr_pred = np.hstack((arr_pred,arr2_pred))
            arr_true = np.hstack((arr_true,arr2_true))
        if hack_idx>limit:
            break
        else:
            hack_idx +=1
        print(velocity, hack_idx,'/',num_files)
    return arr_pred, arr_true

def do_waterfall():
    pred_arr, true_arr = waterfall()
    #pred_arr = waterfall()
    np.savetxt('../../scratch/bd_lstm/teststats/HORpred.csv',pred_arr)
    np.savetxt('../../scratch/bd_lstm/teststats/HORtrue.csv',true_arr)

if __name__ == '__main__':
    do_waterfall()
