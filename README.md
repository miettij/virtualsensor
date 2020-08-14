This repository contains the implementation and demo samples of Approximation of Rotor Vibration with Bidirectional LSTM paper.

This repository is divided into 3 different directories:

1. code
- Includes the files used for preprocessing data and training a bidirectional LSTM network
- To run the demo of the pretrained model that approximates the center-point movements in the time domain:
  - Make sure your environment includes the requirements in requirements.txt
  - move to the directory: "./code/bdrnn/"
  - Run "python3 pretrained_demo.py" on your terminal.

2. original
- For data storing purposes
- master_dataset directory holds the unprocessed raw data files
- minmaxdata directory includes files that can be used to rescale the preprocessed data
- processed_data directory includes data that has been preprocessed from master_dataset

3. scratch
- Includes directories for temporary results. All results should be moved to results directory.
- trainstats subdirectory includes demoweights.pth that have been pretrained. Use them by running pretrained_demo.py -file in the code directory.
