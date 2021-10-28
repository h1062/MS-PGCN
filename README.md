# MS-PGCN
Recently, graph convolutional networks have shown excellent results in skeleton-based action recognition. This paper
presents a Multi-Stage Part-aware Graph Convolutional Network for the problems of model over complication, parameter redun-
dancy and lack of feature information. The structure of this network has a multi-stream input and two-stream output, which can
greatly reduce the complexity of the model and improve the accuracy of the model without losing sequence information. The two
branches of the network have the same backbone, which include 6 multi-order features extraction blocks and 3 temporal attention
calibration blocks, to extract the feature. The output features of two branches are fused from multi-stage of the backbone. In multi-
order features extraction block, a channel-spatial attention mechanism and a graph condensation module are proposed, which can
extract more distinguishable feature and identify the relationship between parts. In temporal attention calibration block, an attention
module is used to calibrate the time frame in the skeleton sequence. Experimental data shows that this model outperforms the
most mainstream methods on NTU and Kinetics datasets. e.g., achieving 92.4 % accuracy on the cross-subject benchmark of
NTU-RGBD60 dataset.

#Framework


![image](https://user-images.githubusercontent.com/75009289/139197712-9707ca85-e69c-43c4-afbc-7a90ad0c398a.png)

Figure 1: Comparisons of different methods on NTU60 (CS setting) in terms of accuracy and the number of parameters. Among these methods, the proposed MS- model achieves the best performance with an order of magnitude smaller model size.
#FRAMEWORK
![image](https://user-images.githubusercontent.com/75009289/139197254-e0313c92-63af-4da4-a4e8-0ca467254a21.png)

Figure2: Illustration of the overall architecture of the MS-PGCN. The scores of two streams are added to obtain the final prediction. In the feature
extraction part, one green block represents an MFEB, and one yellow block represents a TACB.
![image](https://user-images.githubusercontent.com/75009289/139198036-2ae63db9-79f1-42f9-9211-167ad7731196.png)

Figure3:Illustration of the multi-order features extraction block
(MFEB). channel-spatial attention mechanism (CSAM), graph con-
volutional network (GCN) and graph condensation module (GCM).
Notably, GCM only changes the topology and does not change the
number of channels. Moreover, a residual connection is add for every
two GCN.

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
