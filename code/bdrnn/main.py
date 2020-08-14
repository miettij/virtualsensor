
from utils import read_params, filesplit, savefilenames, read_names
from train import train
from dataset import DataSet

DATADIR = '../../original/processed_data/' #normalized cpm (0,1) and force (2...7)
SAVEPATH = '../../scratch/bd_lstm/' # Statistics
cfg_file_path = './conf_model.cfg' # Hyperparameters

def main():
    params = read_params(cfg_file_path)
    trainfiles, testfiles = filesplit(DATADIR)
    savefilenames(SAVEPATH+'filenames/',trainfiles,testfiles)
    trainfiles = read_names('../../scratch/bd_lstm/filenames/trainfiles.txt')
    trainset = DataSet(root_dir = DATADIR, files = trainfiles, normalize = False, seq_len = params['slice_size'], stride = params['stride'])
    train(SAVEPATH+'trainstats/', trainset, params)
main()
