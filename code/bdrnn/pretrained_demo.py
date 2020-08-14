from model import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import types
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import read_names, make_dict, find_closest, read_params, scale_seqs
from dataset import DataSet

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

cfg_file_path = './conf_model.cfg'

def evaluate(TEMPPATH, testfiles, params):

    #Init

    model = LSTM_layers(
        input_size=int(params['input_size']),
        hidden_size=int(params['hidden_size']),
        num_layers=int(params['n_layers']),
        dropout=float(params['dropout']),
        output_size=int(params['output_size']),
        batch_first = True,
        bidirectional = True)
    model.load_state_dict(torch.load(TEMPPATH+'trainstats/weights_middle.pth',map_location=device))
    model.to(device)
    model.eval()
    print_every = 4
    counter = 0
    losses = []
    for testfile in testfiles:
        counter +=1
        lista = [testfile]

        testset = DataSet(root_dir = '../../original/processed_data/', files = lista, normalize = False, seq_len = params['slice_size'], stride = 1000)
        loader = DataLoader(testset, batch_size = int(params['batch_size']), shuffle = True, drop_last = True)


        #For every

        for idx, sample in enumerate(loader):
            y = sample[:,:,:2].clone().detach().requires_grad_(True).to(device)
            x = sample[:,:,2:].clone().detach().requires_grad_(True).to(device)
            h0 = model.init_hidden(int(params['batch_size']),None).to(device)
            c0 = model.init_cell(int(params['batch_size'])).to(device)

            #compute
            loss = F.mse_loss(model.forward(x,(h0,c0)),y)

            #for analytics
            losses.append(loss.item())
            if (idx % print_every) == (print_every - 1):
                print("counter: ",counter,"/",len(testfiles)," mean test loss: ",np.mean(losses))
                break
    f = open(TEMPPATH + 'teststats/testloss.txt','w+')
    f.write("Testloss: "+str(np.mean(losses)))
    f.close()


def evaluate_any_file():
    #os.system(scp )
    filepath = '../../original/processed_data/'
    weightpath = '../../scratch/bd_lstm/trainstats/weights_middle.pth'
    demoweights = '../../scratch/bd_lstm/trainstats/demoweights.pth'
    weightpath = demoweights
    parampath = '../../code/bdrnn/conf_model.cfg'
    filenamepath = '../../scratch/bd_lstm/filenames/testfiles.txt'
    minmaxdatapath = '../../original/minmaxdata/'

    #get best file
    filenames = read_names(filenamepath)
    print("Available files: ")
    for file in filenames:
        print(file)
    filenamedict = make_dict(filenames)
    velocity = float(input('Give rotational velocity between 4Hz and 18Hz and the closest one is used at evaluation.\n'))
    filename, velocity = find_closest(filenamedict, velocity)
    files = [filename]

    #read parameters
    params = read_params(parampath)

    #init dataset with the file we selected and model
    dataset = DataSet(root_dir = filepath, files = files, normalize = False, seq_len = params['slice_size'], stride = 1000)

    loader = DataLoader(dataset, batch_size = int(params['batch_size']), shuffle = True)

    model = LSTM_layers(
        input_size=int(params['input_size']),
        hidden_size=int(params['hidden_size']),
        num_layers=int(params['n_layers']),
        dropout=float(params['dropout']),
        output_size=int(params['output_size']),
        batch_first = True,
        bidirectional = True)
    #RuntimeError: Attempting to deserialize object on a
    #CUDA device but torch.cuda.is_available() is False.
    #If you are running on a CPU-only machine,
    #please use torch.load with map_location='cpu' to map your storages to the CPU.

    model.load_state_dict(torch.load(weightpath, map_location = 'cpu'))
    model.to(device)
    model.eval()
    losses = []

    for idx, sample in enumerate(loader):
        y = sample[:,:,:2].clone().detach().requires_grad_(True).to(device)
        x = sample[:,:,2:].clone().detach().requires_grad_(True).to(device)
        h0 = model.init_hidden(int(params['batch_size']),None).to(device)
        c0 = model.init_cell(int(params['batch_size'])).to(device)

        #compute
        output = model.forward(x,(h0,c0))
        loss = F.mse_loss(output,y)
        losses.append(loss.item())

        output,y = scale_seqs(output,y, filename, minmaxdatapath)

        if (idx%3) == 0 :
            save_this_plot(0, 2763, output[0],y[0],loss.item(),velocity)
    print("Avg loss:",np.mean(losses))

def get_freq_domain(darr):
    n = len(darr)
    k = np.arange(n)
    T = n/int(2000)

    frq = k/T
    frq = frq[range(int(n/50))]

    Y = 2*np.fft.fft(darr)/n
    Y = Y[range(int(n/50))]
    return frq, Y

def get_freq_domain2(darr,vel):
    n = len(darr)
    k = np.arange(n)
    T = n/int(2000/vel)
    frq = k/T
    frq = frq[range(int(n/50))]

    Y = 2*np.fft.fft(darr)/n
    Y = Y[range(int(n/50))]
    return frq, Y

def save_this_plot(start, stop, output, target, loss, speed):
    SAVEFOLDER = '../scratch/plots/' # hard code to GET location
    SAVE_PATH = SAVEFOLDER + str(loss)+'.png'
    output = output.detach().squeeze(1).cpu().numpy()
    target = target.detach().squeeze(1).cpu().numpy()
    t = np.linspace(start/2000,stop/2000, stop-start)
    fig, axes = plt.subplots(4, figsize=(10,10))
    pred_labels = ['Horizontal', 'Vertical']

    for i in range(2): #modify to only draw cpm
        l1, = axes[i].plot(t,output[:,i], label = 'Approximated', color = 'red')
        l2, = axes[i].plot(t,target[:,i], label = 'Measured',color = 'blue')
        axes[i].title.set_text(pred_labels[i])
        plt.legend(handles=[l1,l2])
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Displacement (mm)')

    x_pred = output[:,0]
    y_pred = output[:,1]

    x_true = target[:,0]
    y_true = target[:,1]

    #n = len(x_pred)
    if not (len(x_pred) == len(x_true)):
        print("xpred len :", len(x_pred))
        print("x_true len: ", len(x_true))

    frq, Y = get_freq_domain(x_pred)
    l3, = axes[2].plot(frq,np.abs(Y), color = 'red')
    axes[2].title.set_text(pred_labels[0])
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Amplitude (mm)")

    frq, Y = get_freq_domain(x_true)
    l4, = axes[2].plot(frq,np.abs(Y), color = 'blue')
    axes[3].title.set_text(pred_labels[1])
    axes[3].set_xlabel("Frequency (Hz)")
    axes[3].set_ylabel("Amplitude (mm)")

    frq, Y = get_freq_domain(y_pred)
    l5, = axes[3].plot(frq,np.abs(Y), color = 'red')

    frq, Y = get_freq_domain(y_true)
    l6, = axes[3].plot(frq,np.abs(Y), color = 'blue')

    #plt.subplots_adjust(top=0.85, bottom = -0.85)
    fig.tight_layout()
    print("Rotation speed {}Hz, MSE Loss: {:6f}".format(speed,loss))
    plt.show()



if __name__ == '__main__':
    evaluate_any_file()
