
from utils import read_names
from evaluate import *

def main():
    params = read_params(cfg_file_path)
    TESTIOPATH = '../../scratch/bd_lstm/'
    testfiles = read_names('../../scratch/bd_lstm/filenames/testfiles.txt')
    evaluate(TESTIOPATH,testfiles,params)
main()
