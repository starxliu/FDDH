
################ Information ################

Matlab demo code for "FDDH: Fast Discriminative  Discrete  Hashing for Large-Scale Cross-Modal Retrieval"  accepted by IEEE Transactions on Neural Networks and Learning Systems
Authors: Xin Liu, Xingzhi Wang, and Yiu-ming Cheung;
Contact: xliu[at]hqu.edu.cn


This code uses some public software packages by the 3rd party applications, and is free for educational, academic research and non-profit purposes. Not for commercial/industrial activities. If you use/adapt our code in your work (either as a stand-alone tool or as a component of any algorithm), you need to appropriately cite our work.



################ Tips ################
1. To run a demo, see the FDDH_main package and conduct the following command:
        trainFDDH.m

* If you have got any question, please do not hesitate to contact us.
* Bugs are also welcome to be reported.

################ Contents ################
This package contains cleaned up codes for the FDDH, including:

trainFDDH.m: test example on public Wiki dataset
solveFDDH.m: function to optimize the objective function of FDDH
bitCompact.m: function to compute the compact hash code matrix
hammingDist.m: function to compute the hamming distance between two sets
kernelMatrix.m: function to compute a kernel matrix
kernelTrans: function to do kernel transformation
centerlizeData: function to centerlize the data
calcPreRecRadiusLabel.m: calculate precision and recall within different radius based on Label
calcMapTopkMapTopkPreTopkRecLabel.m: function to obtain the retrieval results.


If you use our codes, we are appreciated if you appropriately cite our work.
################ Citation ################
Xin Liu, Xingzhi Wang and Yiu-ming Cheung; "FDDH: Fast Discriminative Discrete Hashing for Large-Scale Cross-Modal Retrieval", IEEE Transactions on Neural Networks and Learning Systems, doi:10.1109/TNNLS.2021.3076684

