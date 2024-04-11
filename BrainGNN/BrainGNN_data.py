#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:02:10 2024

@author: qqw
"""
#please input your to-be-analyzed data in full_data,including fMRI time series with shape of (nsub,nlength,nroi) and label(y) with the shape of (nsub,).
###################################Data preprocessing###################################


import os
import warnings
import glob
import csv
import re
import numpy as np
import scipy.io as sio
import sys
from nilearn import connectome
import pandas as pd
from scipy.spatial import distance
from scipy import signal
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
import deepdish as dd
import scipy.io
import pickle
with open('/home/qqw/Unsupervised_Pretraining/combine_ADHD_ABIDE_MDD_3806subj_TP170_data.pkl', 'rb') as file:
    full_data = pickle.load(file)  # shape  (3806,170,116)
#full_data1=full_data.tolist()
timeseries1 = []
timeseries2 = []
for i in range(len(full_data)):
    # print(A[i].T.shape)
    
    length = int(full_data.shape[1] * 0.9)
    series1 = full_data[i][:length, :]
    series2 = full_data[i][-length:, :]
    timeseries1.append(series1)
    timeseries2.append(series2)

subject_IDs = [str(num) for num in range(1, 3807)]
#print(subject_IDs)
# y=np.load('/home/qqw/Unsupervised_Pretraining/combine_ADHD_ABIDE_MDD_3806subj_TP170_fileID.npy')
# y[y==0]=2
y=np.ones(3806)

def subject_connectivity(timeseries, kind):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subjects     : subject IDs
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        iter_no      : tangent connectivity iteration number for cross validation evaluation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder
    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    if kind in ['TPE', 'TE', 'correlation','partial correlation']:
        if kind not in ['TPE', 'TE']:
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform(timeseries)#(1035,200,200)
        else:
            if kind == 'TPE':
                conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                conn_mat = conn_measure.fit_transform(timeseries)
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(conn_mat)
                connectivity = connectivity_fit.transform(conn_mat)
            else:
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(timeseries)
                connectivity = connectivity_fit.transform(timeseries)
                
    return connectivity

# ##timeseries 1
v1_connectivity1=subject_connectivity(timeseries1,'correlation')
v1_fea_corr=v1_connectivity1
v1_connectivity2=subject_connectivity(timeseries1,'partial correlation')
v1_fea_pcorr=v1_connectivity2


def create_folder(directory):
    try:
        # Create target Directory
        os.makedirs(directory)  # Use makedirs to create parent directories if they don't exist
        print("Directory", directory, "created")
    except FileExistsError:
        print("Directory", directory, "already exists")

# Specify the directory path you want to create
new_folder_path = "./V1/raw/"

# Call the function to create the folder
create_folder(new_folder_path)
#subject_IDs=['1','2','3']
data_folder='./V1/raw/'
for i, subject in enumerate(subject_IDs):
      dd.io.save(os.path.join(data_folder, subject+'.h5'),{'corr':v1_fea_corr[i],'pcorr':v1_fea_pcorr[i],'label':y[i]%2})


