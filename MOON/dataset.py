from sklearn.model_selection import StratifiedKFold
from torch import tensor,float32
from random import shuffle
import numpy as np
from torch.utils.data import Dataset

class Load_Data(Dataset):
    def __init__(self, data_path, label_path, k_fold=None):
        # data_path: path to the sample file, label_path: path to the label file
        # k_fold: the number of folds, 'None' indicates no folding
        self.data_dict = {}

        self.data = np.load(data_path)
        self.label = np.load(label_path) # labels should consist of 0 and 1

        for id in range(self.data.shape[0]):
            self.data_dict[id] = self.data[id, :, :]

        self.full_subject_list = list(self.data_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
        self.k = None

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    # set folds
    def set_fold(self, fold, train=True):
        # fold: the fold to use
        # train: 'True' indicates using the training set, 'False' indicates using the validation set.
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.label))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [self.full_subject_list[idx] for idx in test_idx]

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        X = self.data_dict[subject]
        y = self.label[subject]

        return {'id': subject, 'X': tensor(X, dtype=float32), 'y': y}

