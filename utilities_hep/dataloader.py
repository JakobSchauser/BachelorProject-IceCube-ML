import numpy as np


import tensorflow as tf
import scipy as sp
import pickle
import gzip

import sqlite3
import os
from pandas import read_sql
import pandas as pd

import sys
args = sys.argv


from sklearn.neighbors import kneighbors_graph as knn
from sklearn.preprocessing import RobustScaler
import spektral 
from spektral.data import Dataset, Graph
from tqdm import tqdm


class LoadParticleDataset(Dataset):
    def __init__(self,data = []):
        self.data = np.array(data)
        
    def read(self):
        return self.data
    
    def generate_data(self,limit = 1e3,_type = "muon", train_test= "train", n_neighbors = 15):
       
        metapath = None
    

        if _type == "muon":
            path = "/groups/hep/pcs557/databases/dev_level2_mu_tau_e_muongun_classification_wnoise/data/dev_level2_mu_tau_e_muongun_classification_wnoise_unscaled.db"
            metapath = "/groups/hep/pcs557/databases/dev_level2_mu_tau_e_muongun_classification_wnoise/data/meta/sets.pkl"
            pid = 13
        elif _type == "neutrino":
            path = "/groups/hep/pcs557/databases/IC8611_oscNext_003_final/data/IC8611_oscNext_003_final_unscaled.db"
        elif _type == "muon_neutrino":
            pid = 14
            path = "/groups/hep/pcs557/databases/dev_lvl7_mu_nu_e_classification_v003/data/dev_lvl7_mu_nu_e_classification_v003_unscaled.db"
            metapath = "/groups/hep/pcs557/databases/dev_lvl7_mu_nu_e_classification_v003/meta/sets.pkl"
        elif _type == "all":
            path = "/groups/hep/pcs557/databases/dev_lvl7_mu_nu_e_classification_v003/data/dev_lvl7_mu_nu_e_classification_v003_unscaled.db"
            metapath = "/groups/hep/pcs557/databases/dev_lvl7_mu_nu_e_classification_v003/meta/sets.pkl"

        if metapath:
            meta = pd.read_pickle(metapath)
            evnt = meta[train_test]["event_no"]


        with sqlite3.connect(path) as con:
            evnt = tuple(evnt)
            print(len(evnt))
            if(_type != "all"):
                query = f'select * from truth where event_no in {evnt} AND pid = {pid} LIMIT {limit}'
            else:
                query = f'select * from truth where event_no in {evnt} LIMIT {limit} ORDER BY random();'



            truth = pd.read_sql(query, con)
        

            # Make sure the same events are used:
            events = tuple(truth["event_no"]) 
            query = f'select * from features where event_no in {events}'  # and SRTInIcePulses = {cleaning}'

            features = pd.read_sql(query, con)

            

        print("Features:",features.columns)
        print("Truths:",truth.columns)
        featurelist = ["dom_x","dom_y","dom_z","dom_time","charge_log10","width","pmt_area"]
        pos = ["dom_x","dom_y","dom_z"]
        target_features = ['energy_log10','azimuth','zenith',"event_no"]
#         target_features = ["stopped_muon"]
#         target_features = ["energy_log10"]
        
        features_arr = np.array(features[featurelist])
        truth_arr = np.array(truth[target_features])

        graphs = []
        targets = []

        # #Find when event type changes
        # _, changes = np.unique(features["event_no"].values,return_index = True)
        # changes = np.append(changes,len(features))
        
        for i in tqdm(range(len(truth_arr))):
            # ind0,ind1 = changes[i],changes[i+1]
            # seq = features[ind0:ind1]
            # target = truth_arr[i]
            eno = truth.iloc[i]["event_no"]
            target = truth_arr[i]
            seq = features[features["event_no"] == eno]

            dim = seq.shape[0] 
            if (dim > n_neighbors):
                nbs = knn(seq[pos],n_neighbors)
            else:
                if dim <= 0:
                    continue
                nbs = np.ones((dim,dim))
                nbs = sp.sparse.csr_matrix(nbs)
                #I have changed this to fully connect properly
                
#             dists = knn(seq[pos],n_neighbors,mode = "distance")
            if len(target) == 0:
              continue
            # print((target.values))
            # print((seq[features]))
            x = np.array(seq[featurelist])
            graph = Graph(x = x, a = nbs.T, y = target)
            
            graphs.append(graph)
            targets.append(target)
        self.data = np.array(graphs,dtype=object)
    def read(self):
        return np.array(self.data)

print("File opened")

