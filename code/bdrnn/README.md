This document explains the functionalities of the files in this directory.

main.py
- Randomly divides files in the dataset located in original/processed_data to training and test files. Filenames are stored in scratch/filenames.
- Trains the bidirectional LSTM model with the training files in original/processed_data.
- Some processed data has been included in this repository.
- Full training dataset can be found from: [Link will be inserted]

model.py
- Defines the neural network class

conf_model.cfg
- The hyperparameters for the model and the training algorithms

train.py
- Defines the training algorithm

test.py
- Computes average test loss over test files.

dataset.py
- Defines the dataset class

evaluate.py
- Includes the following functions:
  - evaluate()
    - computes the average test loss over some dataset
  - evaluate_any_file()
    - Computes loss and plots a random subsequence from a test file including data
      measured close to chosen rotating speed.
  - get_freq_domain()
    - returns the sequence in the frequency domain
  - get_freq_domain2()
    - returns the sequence in the harmonic frequency domain
  - save_this_plot()
    - plots the sequences in the time and frequency domain.
    - The sequences include measured and approximated center-point movements \n
    in the horizontal and vertical directions

gpu_run.sh
- bash script for training the model

compute_waterfall.py
- Compute 3D-arrays in the frequency domain out of the approximated and measured center-point movements

plot_waterfall.py
- plots the 3D-arrays in a 3D-grid

utils.py
- Utility functions
