# ACTION-Open-Source-Software-for-FMRI-Analysis

The Augmentation and Computation Toolbox for braIn netwOrk aNalysis (ACTION) is an open-source Python software, designed for fMRI data augmentation, brain network construction and visualization, extraction of brain network features, and intelligent analysis of brain networks based on AI models. 
Through a graphics user interface, the ACTION aims to provide users with comprehensive, convenient, and easy-to-use fMRI data analysis services, helping users simplify the processing pipeline and improve work efficiency.

Authors: Yuqi Fang, Junhao Zhang, Linmin Wang, Qianqian Wang, Mingxia Liu

Software Download: https://mingxia.web.unc.edu/action/

# 1. FMRI Data Augmentation
## 1.1. BOLD Signal Augmentation 

1. Upsampling
2. Downsampling
3. Slicing
4. Noise Jittering
 
## 1.2. Graph Data Augmentation

1. Random Node Dropping
2. Hub-Preserving Node Dropping
3. Random Edge Perturbation
4. Weight-Dependent Edge Removal
5. Subgraph Cropping

# 2. Brain Network Construction

1. Pearson’s Correlation (PC)
2. Mutual Information (MI)
3. Partial Correlation
4. Spearman’s Correlation (SC)
5. High-Order Functional Connectivity (HOFC)
6. Sparse Representation (SR)
7. Low-rank Representation (LR)

# 3. Brain Network Feature Extraction

1. Node level feature extraction
2. Graph level feature extraction
   
# 4. Artificial Intelligence Model Construction

## 4.1. Machine Learning Model Construction 
  
### Machine Learning Model

1. Support Vector Mahince (SVM)
2. Support Vector Regression (SVR)  
3. Random Forest (RF) 
4. Extreme Gradient Boosting (XGBoost)  
5. K-Nearest Neighbors (KNN)

### Data Partition
1. K-fold Cross Validation
2. Random Partition

### Dimension Reduction
1. Principle Component Analysis (PCA)
2. Canonical Correlation Analysis (CCA)
3. Independent Component Analysis (ICA)

## 4.2. Deep Learning-based Foundation Model Construction

### Overview

This module focuses on (1) pretraining foundation models that can easily adapt to downstream tasks for fMRI analysis, and (2) implementing popular federated learning methods to facilitate multi-site fMRI studies. 
1. The feature encoders of ten deep learning models are pretrained on 3,806 unlabeled fMRI scans from public cohorts in a self-supervised learning manner. In each directory, the `Pretrain_XXX.py` and `Finetune_XXX.py` scripts are the main functions for pretraining and fine-tuning a specific encoder, respectively.
2. The source code of five federated learning methods are released to the public, where the users can use the pretrained backbone in this toolbox as the encoder for federated learning. 

### Pretrained Foundation Model

In this project, we pretrain several popular encoders on auxiliary fMRI scans. The pretrained encoders include:

1. Graph Convolutional Network (GCN)
2. Graph Attention Network (GAT)
3. Graph Isomorphism Network (GIN)
4. BrainGNN
5. BrainNetCNN
6. Spatio-Temporal Attention Graph Isomorphism Network (STAGIN)
7. Spatio-Temporal Graph Convolutional Network (STGCN)
8. Graph SAmple and aggreGatE (GraphSAGE) 
9. Transformer
10. Modularity-Constrained Graph Neural Network (MCGNN)

### Federated Learning Strategy
1. Federated Averaging (FedAvg)
2. Federated Proximal (FedProx)
3. Model-contrastive federated learning (MOON)
4. Personalized Federated learning with Moreau envelope (pFedMe)
5. Local Global Federated Averaging (LGFedAvg)

### Usage
You can fine-tune pretrained encoders for various fMRI-based analyses.
Note that the `Finetune_XXX.py` script is just an example of how to finetune our pretrained encoders for classification tasks, and one can modify this code according to different downstream tasks.

Specifically, for 'GCN', 'GIN', 'GAT', 'BrainNetCNN', 'GraphSAGE', 'STAGIN', 'STGCN', 'MGNN', and 'Transformer', 
please input your to-be-analyzed data in the `Data` class of the corresponding `Finetune_XXX.py`,
including fMRI time series with shape of (nsub,nlength,nroi) and label with the shape of (nsub,).

For 'BrainGNN', please input your to-be-analyzed data in `BrainGNN_data.py`,
including fMRI time series with shape of (nsub,nlength,nroi) and label with the shape of (nsub,).

Ensure that your data includes fMRI time series with a shape of (nsub, nlength, nroi) and labels with a shape of (nsub,), where:
- `nsub`: the number of subjects
- `nlength`: the length of fMRI time series
- `nroi`: the number of regions-of-interest (ROIs)
 

Acknowledgments to the following public projects:
[SimSiam](https://github.com/facebookresearch/simsiam),
[UCGL](https://github.com/mxliu/Unsupervised-Contrastive-Graph-Learning),
[GCN](https://github.com/tkipf/gcn)
[GAT](https://github.com/gordicaleksa/pytorch-GAT),
[Transformer](https://github.com/gordicaleksa/pytorch-original-transformer/tree/main),
[BrainGNN](https://github.com/xxlya/BrainGNN_Pytorch),
[BrainNetCNN](https://github.com/nicofarr/brainnetcnnVis_pytorch/tree/master),
[GraphSAGE](https://github.com/williamleif/graphsage-simple),
[STAGIN](https://github.com/egyptdj/stagin),
[STGCN](https://github.com/sgadgil6/cnslab_fmri).


# License
The University of North Carolina at Chapel Hill (UNC-CH) holds all rights to the ACTION software, which is available at no cost for users in academia. 
Individuals are permitted to distribute and modify ACTION under the conditions of the GNU General Public License issued by the Free Software Foundation. 
Commercial or industrial entities interested in utilizing ACTION must reach out to both the University and the tool's author for permission. 
While ACTION is provided with the hope of being beneficial, it comes with no guarantees, including no implied warranties of being sellable or suitable for any specific use. 
Please refer to the GNU General Public License (https://www.gnu.org/licenses/licenses.html#GPL) for further information. 
