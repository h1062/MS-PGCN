
# MS-PGCN
We plan to make further improvements to MS-PGCN. We will be very sorry if it troubles you. I hope to understand, thank you!


#Prerequisites




The code is built with the following libraries:

Python 3.6
Anaconda
PyTorch 1.3
#Data Preparation

We use the dataset of NTU60 RGB+D as an example for description. We need to first dowload the NTU-RGB+D dataset.

Extract the dataset to ./data/ntu/nturgb+d_skeletons/




#Process the data
```
cd ./data/ntu
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

#Training
```
# For the CS setting
python  main.py --network MSPGCN --train 1 --case 0
# For the CV setting
python  main.py --network MSPGCN --train 1 --case 1
```




#Testing




Test the pre-trained models (./results/NTU/MSPGCN/)
```
# For the CS setting
python  main.py --network MSPGN --train 0 --case 0
# For the CV setting
python  main.py --network MSPGN --train 0 --case 1
```




#Contributing



In this paper, a new multi-stage part-aware graph convolutional
network is proposed for skeleton-based action recognition. In MS-
PGCN, the basic block MFEB used to extract spatial feature infor-
mation, and its internal GCM can condense the joints with associated
significance in the topological structure of the thumbnail. Accord-
ing to these functions, the structure of network is multiple-input
and multiple-output. The reason is that as the number of network
layers deepens, the network will gradually ignore the lower-level
local feature information, so the model outputs feature information
in different network layers. To achieve the purpose of saving both
local detailed features and global semantic features. In addition, a
spatiotemporal attention mechanism is embedded in each graph con-
volutional layer to help the model pay more attention to important
joints and frames. This method has a positive effect on increasing the
robustness and generalization of the model. At the same time, TACM
is proposed in order to make full use of temporal high-order seman-
tic information, which aims to filter temporal noise to highlight key
frames in action sequences.
