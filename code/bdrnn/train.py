import torch
from torch.utils.data import DataLoader
from model import LSTM_layers
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def train(TEMPPATH, trainset, params):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    plt.rcParams.update({'font.size': 12})
    start = datetime.now()
    date_time = start.strftime('%m%d%Y,%H:%M')

    loader = DataLoader(trainset, batch_size = int(params['batch_size']), shuffle = True, drop_last = True)

    model = LSTM_layers(
    input_size=int(params['input_size']),
    hidden_size=int(params['hidden_size']),
    num_layers=int(params['n_layers']),
    dropout=float(params['dropout']),
    output_size=int(params['output_size']),
    batch_first = True,
    bidirectional = True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    print_every = 500

    epoch_loss = []
    losses = []
    best_loss  = 10# save trigger
    f = open(TEMPPATH + 'traindata.txt','w+')
    #for i in range(3):
    for i in range(int(params['n_epochs'])):
        running_loss = 0.0

        for idx, sample in enumerate(loader):
            print(idx, "sample: ",sample.shape)
            #training loop init
            y = sample[:,:,:2].clone().detach().requires_grad_(True).to(device)
            x = sample[:,:,2:].clone().detach().requires_grad_(True).to(device)
            h0 = model.init_hidden(int(params['batch_size']),None).to(device)
            c0 = model.init_cell(int(params['batch_size'])).to(device)

            #optimisation
            optimizer.zero_grad()
            loss = F.mse_loss(model.forward(x,(h0,c0)),y)
            loss.backward()

            optimizer.step()

            #analytics
            floss = float(loss.item())
            if np.isnan(loss.item()):
                break
            running_loss += floss
            losses.append(floss)
            epoch_loss.append(floss)
            if (idx % print_every) == (print_every - 1):
                true_loss = running_loss/print_every
                if true_loss < best_loss:
                    best_loss = true_loss

                    torch.save(model.state_dict(), TEMPPATH+'weights_middle.pth')

                    f.write(str("\nsaved best step after: "+str(idx)+" minibatches\n"))
                    f.write(str('Epoch: '+ str(i)+' minibatch: '+str(idx)+" avg_loss:"+ str(true_loss) + '\n'))
                    fig, ax = plt.subplots(1)
                    ax.set_xlabel('Training Batches')
                    ax.set_ylabel('Mean Squared Error')
                    ax.title.set_text('Training Loss')
                    ax.loglog(losses)
                    plt.tight_layout()
                    fig.savefig(TEMPPATH + '_logloss_'+ '.png')


                print("loss: ", true_loss)
                now = datetime.now()-start
                delta = str(now)
                print("time taken for 500 samples ",delta )
                running_loss = 0.0

        now = datetime.now()-start
        delta = str(now)
        print("time taken this epoch ",delta )
        running_loss = 0.0
        f.write(str('Epoch: '+ str(i)+' loss: '+ str(np.mean(epoch_loss)) + '\n'))
        epoch_loss = []
    f.close()
    fig, ax = plt.subplots(1)
    ax.set_xlabel('Training Batches')
    ax.set_ylabel('Mean Squared Error')
    ax.title.set_text('Training Loss')
    ax.loglog(losses)
    plt.tight_layout()
    fig.savefig(TEMPPATH + '_logloss_'+ '.png')
    torch.save(model.state_dict(), TEMPPATH+'weights.pth')
