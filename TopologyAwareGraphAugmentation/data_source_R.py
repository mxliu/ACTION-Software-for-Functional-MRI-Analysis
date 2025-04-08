
###This is edge augmentation#######
import warnings
warnings.simplefilter('ignore')
import scipy.io
import numpy as np
import torch
from einops import rearrange
from sklearn.model_selection import StratifiedKFold
##############original adj and feature###########

# #############################above: construct adj on all sites of ABIDE#####################################


################################following: selecting subjects with time length>=230 on two MDD and ABIDE############
import pickle

# Specify the path to your Pickle file
pickle_file_path = '/home/qqw/Downloads/newfMRI_data/ABIDE_1035_aal/data.pickle'

# Open the Pickle file in binary mode for reading
with open(pickle_file_path, 'rb') as file:
    # Load the object from the Pickle file
     loaded_object = pickle.load(file)#list 1035 each element: (time, 116)

import numpy as np

# Assuming 'your_list' is the input list with ndarrays
your_list = loaded_object # Replace [...] with your actual list

selected_elements = []
selected_indices = []

for index, element in enumerate(your_list):
    if element.shape[0] >= 230:
        selected_elements.append(element[0:230,:])
        selected_indices.append(index)

################################this is data selection for MDD dataset############


import pickle

site1 = scipy.io.loadmat('/home/qqw/Downloads/newfMRI_data/all.mat')
A = np.squeeze(site1['AAL'].T).tolist()

import numpy as np
selected_elements1 = []
selected_indices1 = []

for index, element in enumerate(A):
    if element.shape[0] >= 230:
        selected_elements1.append(element[0:230,:])
        selected_indices1.append(index)

############################ABIDE+MDD as source data#######################
list_all = selected_elements+ selected_elements1
loaded_object=list_all

###########################################above:selecting subjects with time length>=230 on  MDD and ABIDE####################################

series=[]
for i in range(len(loaded_object)):
    signal = loaded_object[i]  # (175,116)
    pc = np.corrcoef(loaded_object[i].T)  # (116,116)
    pc = np.nan_to_num(pc)
    pc = abs(pc)
    series.append(pc)
adj = np.array(series) #(1035,116,116)



DC_list=[]#184 each 116
for i in range(len(adj)):
    dc = np.sum(adj[i], axis=1).tolist()
    dc = [x + 1e-10 for x in dc] # to avoid too much nodes whose degree is zero
    DC_list.append(dc)

adj=torch.tensor(adj)
input_adj=adj#(184,116,116)
input_fea=adj




####################some functions for node dropping#############
def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out

def weighted_sampling_without_replacement(data, probabilities, k):
    probabilities = probabilities / np.sum(probabilities)

    indices = np.random.choice(len(data), size=k, replace=False, p=probabilities)

    samples = [data[i] for i in indices]

    return samples
def random_sampling(data,k):
    indices = np.random.choice(len(data), size=k, replace=False)

    samples = [data[i] for i in indices]

    return samples
def node_dropping(input_fea, input_adj, DC,remain_percent=0.9):
    node_num = input_fea.shape[0]
    all_node_list = [i for i in range(node_num)]#[0,1,2..115]
    k = int(len(all_node_list) * remain_percent)  #104
    selected_roi = sorted(weighted_sampling_without_replacement(all_node_list, DC, k))#list114
    drop_node_list =sorted(list(set(all_node_list)-set(selected_roi)))#list12
    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)##(104,104)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)#(104,116)
    #tensor  (2348,1433) (2348,2348)
    # aug_input_fea = aug_input_fea.unsqueeze(0)
    # aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj



def random_node_dropping(input_fea, input_adj, DC,remain_percent=0.9):
    node_num = input_fea.shape[0]
    all_node_list = [i for i in range(node_num)]#[0,1,2..115]
    k = int(len(all_node_list) * remain_percent)
    selected_roi = sorted(random_sampling(all_node_list,  k))#list114
    drop_node_list =sorted(list(set(all_node_list)-set(selected_roi)))#list12
    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)##(104,104)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)#(104,116)
    #tensor  (2348,1433) (2348,2348)
    # aug_input_fea = aug_input_fea.unsqueeze(0)
    # aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj
######################################this is hub-preserving node droping (HND) #########################################
class HND(object):
    def read_data(self):
        # aug_edge_adj=edge

        # view1
        aug_fea_list = []
        aug_adj_list = []
        for i in range(input_fea.shape[0]):
           # print(i)
            aug_input_fea, aug_input_adj = node_dropping(input_fea[i], input_adj[i], DC_list[i], remain_percent=0.9)
            aug_fea_list.append(aug_input_fea)
            aug_adj_list.append(aug_input_adj)
        AUG_fea1 = torch.stack(aug_fea_list)
        print(AUG_fea1.shape)  # torch.Size([533, 104, 116])
        AUG_adj1 = torch.stack(aug_adj_list)
        print(AUG_adj1.shape)  # torch.Size([533, 104, 104])

        # view2
        aug_fea_list = []
        aug_adj_list = []
        for i in range(input_fea.shape[0]):
            aug_input_fea, aug_input_adj = node_dropping(input_fea[i], input_adj[i], DC_list[i], remain_percent=0.9)
            aug_fea_list.append(aug_input_fea)
            aug_adj_list.append(aug_input_adj)
        AUG_fea2 = torch.stack(aug_fea_list)
        print(AUG_fea2.shape)  # torch.Size([533, 104, 116])
        AUG_adj2 = torch.stack(aug_adj_list)
        print(AUG_adj2.shape)  # torch.Size([533, 104, 104])

        adj1 = AUG_adj1
        fea1 = AUG_fea1
        adj2 = AUG_adj2
        fea2 = AUG_fea2
        # print(y.shape)#(533,)
        return adj1, fea1, adj2, fea2

    def __init__(self):
        super(HND, self).__init__()
        adj1, fea1, adj2, fea2= self.read_data()

        self.adj1 = adj1
        self.fea1 = fea1
        self.adj2 = adj2
        self.fea2 = fea2
        self.n_samples = adj.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.adj1[index], self.fea1[index],  self.adj2[index], self.fea2[index]

DATA=HND()

######################################this is random node droping (RND) #####################################################
class RND(object):
    def read_data(self):
        # aug_edge_adj=edge

        # view1
        aug_fea_list = []
        aug_adj_list = []
        for i in range(input_fea.shape[0]):
            aug_input_fea, aug_input_adj = random_node_dropping(input_fea[i], input_adj[i], DC_list[i], remain_percent=0.9)
            aug_fea_list.append(aug_input_fea)
            aug_adj_list.append(aug_input_adj)
        AUG_fea1 = torch.stack(aug_fea_list)
        print(AUG_fea1.shape)  # torch.Size([533, 104, 116])
        AUG_adj1 = torch.stack(aug_adj_list)
        print(AUG_adj1.shape)  # torch.Size([533, 104, 104])

        # view2
        aug_fea_list = []
        aug_adj_list = []
        for i in range(input_fea.shape[0]):
            aug_input_fea, aug_input_adj = random_node_dropping(input_fea[i], input_adj[i], DC_list[i], remain_percent=0.9)
            aug_fea_list.append(aug_input_fea)
            aug_adj_list.append(aug_input_adj)
        AUG_fea2 = torch.stack(aug_fea_list)
        print(AUG_fea2.shape)  # torch.Size([533, 104, 116])
        AUG_adj2 = torch.stack(aug_adj_list)
        print(AUG_adj2.shape)  # torch.Size([533, 104, 104])
        adj1 = AUG_adj1
        fea1 = AUG_fea1
        adj2 = AUG_adj2
        fea2 = AUG_fea2
        # print(y.shape)#(533,)
        return adj1, fea1, adj2, fea2

    def __init__(self):
        super(RND, self).__init__()
        adj1, fea1, adj2, fea2= self.read_data()

        self.adj1 = adj1
        self.fea1 = fea1
        self.adj2 = adj2
        self.fea2 = fea2
        self.n_samples = adj.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.adj1[index], self.fea1[index],  self.adj2[index], self.fea2[index]



#################some functions for edge removing###############

def restore_symmetric_matrix(upper_triangular_elements,n):
  # n = int((np.sqrt(1 + 8 * len(lower_triangular_elements)) - 1) / 2)
   # n=4#
    symmetric_matrix = np.zeros((n, n))  #
    index = 0
    for i in range(0,n):
        #print('i value',i)
        for j in range(i+1,n):
          #  print('j value',j)
            symmetric_matrix[i][j] = upper_triangular_elements[index]
            symmetric_matrix[j][i] = upper_triangular_elements[index]
            index += 1
    np.fill_diagonal(symmetric_matrix, 1)
    return symmetric_matrix
def random_sampling(data,k):
    indices = np.random.choice(len(data), size=k, replace=False)

    return indices

def weighted_sampling_without_replacement(data, probabilities, k):
    #
    probabilities = probabilities / np.sum(probabilities)
    # Adjust k if it is larger than the number of non-zero entries
    num_nonzero_entries = len(np.nonzero(probabilities)[0])
    k = min(k, num_nonzero_entries)
    # replace=False
    indices = np.random.choice(len(data), size=k, replace=False, p=probabilities)
    return indices

#FUNCTION OF WER
def edge_perturbation(data, remain_percent):
    upper = data[np.triu_indices(116, 1)]
    k = int(upper.shape[0] * remain_percent)

    #selected_idx = random_sampling(upper.tolist(),  k)
    selected_idx = weighted_sampling_without_replacement(upper, upper, k)#this is ndarray 6003
    all_idx = [i for i in range(upper.shape[0])]
    removed_idx = sorted(list(set(all_idx) - set(selected_idx)))
    for i in range(len(removed_idx)):
        idx = removed_idx[i]
        upper[idx] = 0
    aug_adj = restore_symmetric_matrix(upper, 116)

    return aug_adj


#FUNTION OF RER
def random_edge_perturbation(data, remain_percent):
    upper = data[np.triu_indices(116, 1)]
    k = int(upper.shape[0] * remain_percent)

    selected_idx = random_sampling(upper.tolist(),  k)
    #selected_idx = weighted_sampling_without_replacement(upper, upper, k)#this is ndarray 6003
    all_idx = [i for i in range(upper.shape[0])]
    removed_idx = sorted(list(set(all_idx) - set(selected_idx)))
    for i in range(len(removed_idx)):
        idx = removed_idx[i]
        upper[idx] = 0
    aug_adj = restore_symmetric_matrix(upper, 116)

    return aug_adj


######################################this is weight-dependent edge removing(WER)#########################################
#dataset construction
class WER(object):
    def read_data(self):
        # aug_edge_adj=edge
        input_adj = adj.numpy() #(1035,116,116)
        aug_adj_list1 = []
     #   input_adj=np.delete(input_adj,40,axis=0)
        for i in range(adj.shape[0]):
           # print(i)
            aug_adj = edge_perturbation(input_adj[i], remain_percent=0.9)
            aug_adj_list1.append(aug_adj)
        AUG_adj1 = np.stack(aug_adj_list1)
        print('augmented adj1', AUG_adj1.shape)  #

        # view2
        input_adj = adj.numpy()
        aug_adj_list2 = []
        for i in range(adj.shape[0]):
            aug_adj = edge_perturbation(input_adj[i], remain_percent=0.9)
            aug_adj_list2.append(aug_adj)
        AUG_adj2 = np.stack(aug_adj_list2)
        print('augmented adj2', AUG_adj2.shape)  #

        AUG_fea1 = input_fea
        AUG_fea2 = input_fea
        adj1 = AUG_adj1
        fea1 = AUG_fea1
        adj2 = AUG_adj2
        fea2 = AUG_fea2
        # print(y.shape)#(533,)
        return adj1, fea1, adj2, fea2

    def __init__(self):
        super(WER, self).__init__()
        adj1, fea1, adj2, fea2= self.read_data()

        self.adj1 = adj1
        self.fea1 = fea1
        self.adj2 = adj2
        self.fea2 = fea2
        self.n_samples = adj.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.adj1[index], self.fea1[index],  self.adj2[index], self.fea2[index]

#train_dataset=WER()

######################################this is random edge removing(RER)#########################################

class RER(object):
    def read_data(self):
        # aug_edge_adj=edge
        input_adj = adj.numpy()
        aug_adj_list1 = []
        for i in range(adj.shape[0]):
            aug_adj = random_edge_perturbation(input_adj[i], remain_percent=0.9)
            aug_adj_list1.append(aug_adj)
        AUG_adj1 = np.stack(aug_adj_list1)
        print('augmented adj1', AUG_adj1.shape)  #

        # view2
        input_adj = adj.numpy()
        aug_adj_list2 = []
        for i in range(adj.shape[0]):
            aug_adj = random_edge_perturbation(input_adj[i], remain_percent=0.9)
            aug_adj_list2.append(aug_adj)
        AUG_adj2 = np.stack(aug_adj_list2)
        print('augmented adj2', AUG_adj2.shape)  #

        AUG_fea1 = input_fea
        AUG_fea2 = input_fea
        adj1 = AUG_adj1
        fea1 = AUG_fea1
        adj2 = AUG_adj2
        fea2 = AUG_fea2
        # print(y.shape)#(533,)
        return adj1, fea1, adj2, fea2

    def __init__(self):
        super(RER, self).__init__()
        adj1, fea1, adj2, fea2= self.read_data()

        self.adj1 = adj1
        self.fea1 = fea1
        self.adj2 = adj2
        self.fea2 = fea2
        self.n_samples = adj.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.adj1[index], self.fea1[index],  self.adj2[index], self.fea2[index]


