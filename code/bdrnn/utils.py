import numpy as np
from os import listdir
import pickle

def read_params(filepath):
    """Reads configuration files and returns hyperparameters in a dictionary."""
    param_dict = {}
    f = open(filepath, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.split(';')
        line[1].strip('\n')
        param_dict[line[0]] = line[1]
    f.close()
    return param_dict

def savefilenames(TEMPPATH,trainfiles, testfiles):
    """Saves training and testing filenames to trainfiles.txt and testfiles.txt correspondingly"""
    f = open(TEMPPATH + 'trainfiles.txt','w+')
    for trainfile in trainfiles:
        f.write(trainfile+'\n')
    f.close()
    f = open(TEMPPATH + 'testfiles.txt','w+')
    for testfile in testfiles:
        f.write(testfile+'\n')
    f.close()

def filesplit(root_dir):
    """Splits filenames in the directory into two distinct lists:
    - trainingfiles
    - test files """
    files = listdir(root_dir)
    trainset = np.random.choice(files, size = int(len(files)*0.66), replace = False).tolist()
    testset = [x for x in files if x not in trainset]

    return trainset, testset

def read_names(filenamepath):
    "Reads all filenames in the file pointed by filenamepath and returns them in a list"
    f = open(filenamepath,'r')
    lines = f.readlines()
    filenamelist = []
    for line in lines:
        if '.DS_Store' not in line:
            filenamelist.append(line.strip('\n'))

    return filenamelist

def make_dict(filenamelist):
    """Returns a dictionary of filenames.
    Dictionary key is the rotating speed"""
    filenamedict = {}
    for filename in filenamelist:
        filenameparts = filename.strip('.csv').split('_')
        key = float(filenameparts[-1].strip('Hz'))

        #key = float(filename[12:].strip('.csv'))
        filenamedict[key] = filename
    return filenamedict

def find_closest(filenamedict, velocity):
    "Returns the file including data measured at rotating speed close to 'velocity' "
    keys = filenamedict.keys()
    best_key = 20000
    for key in keys:
        if np.abs(velocity-key) < np.abs(velocity-best_key):
            best_key = key
    return filenamedict[best_key], best_key

def get_scaling_factors(old_filename,minmaxdatapath):
    """Returns scaling factors for the center-point movement."""
    fpath = minmaxdatapath+'/'+old_filename
    with open(fpath, 'rb') as f:
        lib = pickle.load(f)
    x = lib['x']
    y = lib['y']
    return x,y

def scale_signal(signal,scalex,scaley):
    "Rescales the center-point movement signals to the original amplitudes."
    #minibatch part 1: signal[0]

        #horizontal: signal[0,:,0]
    signal[0,:,0] = scale_signal_1d(signal[0,:,0],scalex[0],scalex[1])

        #vertical: signal[0,:,1]
    signal[0,:,1] = scale_signal_1d(signal[0,:,1],scaley[0],scaley[1])

    #minibatch part 2: signal[1]
        #horizontal: signal[1,:,0]
    signal[1,:,0] = scale_signal_1d(signal[1,:,0],scalex[0],scalex[1])
        #vertical: signal[1,:,1]
    signal[1,:,1] = scale_signal_1d(signal[1,:,1],scaley[0],scaley[1])
    return signal

def scale_signal_1d(signal,minv,maxv):
    "Rescales a signal to the original scale."
    signal = (signal+1)/2*(maxv-minv)+minv
    return signal

def scale_seqs(output, ref, filename, minmaxdatapath):
    """Function for scaling the approximated and measured center point movements
    to the original scale."""
    minmaxdatapath = '../../original/minmaxdata'
    fnp = filename.split('_') #filenameparts
    old_filename = fnp[0]+'_'+fnp[1]+'_'+fnp[2]+'_'+fnp[3]+'mm.lvm'
    datadirectory = listdir(minmaxdatapath)
    if old_filename in datadirectory:
        #print(old_filename,'in datadirre')
        scalex, scaley = get_scaling_factors(old_filename,minmaxdatapath)
        scaled_output = scale_signal(output,scalex,scaley)
        scaled_ref = scale_signal(ref,scalex,scaley)
    else:
        raise NameError('could not find: '+old_filename)

    return scaled_output, scaled_ref

def ascendingorder_wf(filenamelist):
    speeds = []
    print(len(filenamelist))
    for filename in filenamelist:
        filename = filename.strip('.csv')
        fnparts =  filename.split('_')
        speed = fnparts[-1].strip('Hz')
        speeds.append(float(speed))

    return speeds

def get_freq_domain(darr):
    "Returns the given signal in frequency domain"
    n = len(darr)
    k = np.arange(n)
    T = n/int(2000)
    frq = k/T
    frq = frq[range(int(n/50))]
    Y = 2*np.fft.fft(darr)/n
    Y = Y[range(int(n/50))]
    return frq, Y

def get_freq_domain2(darr,vel):
    "Returns the given signal in the harmonic frequency domain"
    n = len(darr)
    k = np.arange(n)
    T = n/int(2000/vel)
    frq = k/T
    frq = frq[range(int(n/50))]
    Y = 2*np.fft.fft(darr)/n
    Y = Y[range(int(n/50))]
    return frq, Y
