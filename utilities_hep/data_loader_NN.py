import sqlite3, os, sys, pickle, tqdm, shutil
from sqlite3.dbapi2 import enable_shared_cache

import os.path as osp
import tensorflow as tf
import numpy as np
from multiprocessing import Process, Pool

from pandas import read_sql, read_pickle

from spektral.data import Dataset, Graph
from scipy.sparse import csr_matrix

verbose = True


file_path = osp.dirname(osp.realpath(__file__))

def instructions_to_dataset_name(construction_dict):
    Data         = construction_dict['Data']
    GraphType    = construction_dict['GraphType']
    GraphParam   = construction_dict['GraphParam']

    name = Data + "_" + GraphType + str(GraphParam)

    return name 

def remove_dataset(Data, GraphType, GraphParam = None):
    # function to remove data_folder and adjacency completely to restart

    # Remove data
    Data_folder = osp.join(file_path, "data", "features")
    if Data in os.listdir(Data_folder):
        shutil.rmtree(osp.join(Data_folder, Data))

    # Remove adjacancy folder
    Graph_Name = Data + "_" + GraphType + str(GraphParam)
    Adjancancy_folder = osp.join(file_path, "data", "adjacency")

    if Graph_Name  in os.listdir(Adjancancy_folder):
        shutil.rmtree(osp.join(Adjancancy_folder, Graph_Name))


def split_events(ids, data_splits = [0.8, 0.1, 0.1], seed = 25):
    # Setup seed
    np.random.seed(seed)
    
    # permutate the indices of event numers
    N = len(ids)
    idxs = np.random.permutation(N)
    train_split = int(data_splits[0] * N)
    val_split   = int(data_splits[1] * N) + train_split

    # Split indices
    idx_tr, idx_val, idx_test  = np.split(idxs, [train_split, val_split])

    # Split events
    train_events = ids[idx_tr]
    val_events   = ids[idx_val]
    test_events  = ids[idx_test]

    return train_events, val_events, test_events


def check_dataset(Data, GraphType, GraphParam = None):
    """
    Check if a given dataset is generated, else initiate the process
    Return data_exists, as_exists
    Boolean determing if x data file and as data file are constructed
    """
    Data_folder = osp.join(file_path,  "data", "features")
    if Data not in os.listdir(Data_folder):
        os.mkdir(osp.join(Data_folder, Data))
        data_exists = False
    else:
        data_exists = True
    
    Graph_Name = Data + "_" + GraphType + str(GraphParam)

    Adjancancy_folder = osp.join(file_path,  "data", "adjacency")

    if Graph_Name not in os.listdir(Adjancancy_folder):
        os.mkdir(osp.join(Adjancancy_folder, Graph_Name))
        as_exists    = False
    else:
        as_exists    = True

    return data_exists, as_exists


def neighbors(values, self_loop = False, k = 6):
    # Returns a N x 2 array, where N is the amount of connections
    # Pairs each node with up to k // 2 neighbors on both sides

    sorted_idxs  = np.argsort(values)

    N_nodes      = len(sorted_idxs)

    side_band    = k // 2

    sparse_idx   = []

    for i in range(-side_band, side_band + 1):
        if i == 0 and self_loop == False:
            continue
        else:
            idxs = np.arange(max(0, i), min(N_nodes, N_nodes - i))
            sparse_idx.append(np.vstack([sorted_idxs[idxs], np.roll(sorted_idxs, i)[idxs]]))

    sparse_idx = np.column_stack(sparse_idx).T
    return sparse_idx.astype(int)

file_path = osp.dirname(osp.realpath(__file__))



class graph_dataset(Dataset):
    """
    General Dataset Graph given arguments in json file
    """


    def __init__(self, construct_dict, type = "train", initialize = False):
        # Initialize the Dataset, mostly just unpack the construction dictionairy
        self.type        = type

        self.name        = instructions_to_dataset_name(construct_dict)

        self.Data        = construct_dict['Data']
        self.GraphType   = construct_dict['GraphType']
        self.GraphParam  = construct_dict['GraphParam']

        self.raw_path    = construct_dict['raw_path']

        self.event_lims  = construct_dict['event_lims']
        self.node_lims   = construct_dict['node_lims']

        self.graph_batch = construct_dict['graph_batch']
        self.buffer_size = construct_dict['buffer_size']
        self.data_split  = construct_dict['data_split'] 
        if "max_split" in construct_dict.keys():
            self.max_split = construct_dict["max_split"]
        else:
            self.max_split = None

        self.seed        = 25
        
        self.features    = construct_dict["features"]
        self.targets     = construct_dict["targets"]
        self.transform   = construct_dict["transforms"]

        self.verbose     = construct_dict["verbose"]

        if construct_dict["clear_dataset"] and initialize:
            if construct_dict['verbose']:
                print("Redoing dataset")
            remove_dataset(self.Data, self.GraphType, self.GraphParam)

        super().__init__()


    @property
    def path(self):
        return osp.join(file_path, "data", "adjacency", self.name)


    def download(self):
        if self.verbose:
            print("Preparing dataset")

        # Check if the data exists, if not create directories
        xs_exists, as_exists = check_dataset(self.Data, self.GraphType, self.GraphParam)
        print(xs_exists,as_exists)
        if not self.raw_path:
            self.raw_path = osp.join(file_path, "data", "raw", self.Data + ".db")
        
        x_path = osp.join(file_path, "data", "features", self.Data)
        a_path = self.path

        A_func = "neighbors"

        if self.transform:
            transformers = pickle.load(open(self.transform, "rb"))

        with sqlite3.connect(self.raw_path) as conn:    # Connect to raw database    
            if self.verbose:
                print(f"Connected to {self.raw_path}")
            # Gather ids from sql file
            event_query = "select event_no from truth"
            if self.event_lims:
                event_query += " where " + self.event_lims
            event_ids = np.array(read_sql(event_query, conn)).flatten()
            # print(event_ids)
            
            # Split event_numbers in train/test
            if type(self.data_split) == str:
                sets = read_pickle(self.data_split)
                # train_events, val_events     = list(sets['train'].event_no), list(sets['test'].event_no)
                # test_events                  = list(sets['test'].event_no)
                train_events, val_events     = list(sets['train']), list(sets['test'])
                test_events                  = list(sets['test'])
                
            else:
                train_events, val_events, test_events = split_events(event_ids, self.data_split, self.seed)

            np.random.shuffle(train_events)
            np.random.shuffle(val_events)
            np.random.shuffle(test_events)

            del event_ids # Remove unecessary ram usage
            if self.max_split:
                train_events = train_events[:self.max_split[0]]
                val_events   = val_events[:self.max_split[1]]
                test_events  = test_events[:self.max_split[2]]
            

            # Generate x features if they do not exist
            if not xs_exists:
                
                # Loop over train, validation and test
                for data_type, events in zip(["train", "val", "test"], [train_events, val_events, test_events]):
                    
                    if verbose:
                        print(f"Extracting features for {data_type}")

                    # generate x features loop
                    for i in tqdm.tqdm(range(0, len(events), self.graph_batch)):
                        get_ids = events[i: i + self.graph_batch]

                        # print(get_ids)

                        feature_query = f"select event_no, {', '.join(self.features)} from features where event_no in {tuple(get_ids)}"
                        if self.node_lims:
                            feature_query += " and " + self.node_lims # Add further restrictions

                        features      = read_sql(feature_query, conn).sort_values('event_no')

                        target_query = f"select {'event_no, ' + ', '.join(self.targets)} from truth where event_no in {tuple(get_ids)}"

                        targets      = read_sql(target_query, conn).sort_values('event_no')

                        # Convert to np arrays and split xs in list
                        f_event      = np.array(features['event_no'])
                        x_long       = np.array(features[self.features])
                        ys           = np.array(targets)


                        if self.transform:
                            for col, trans in enumerate(self.features):
                                if trans in list(transformers['features'].keys()):
                                    x_long[:, col] = transformers["features"][trans].inverse_transform(x_long[:, col].reshape(-1, 1)).flatten()
                            for col, trans in enumerate(["event_no"] + self.targets):
                                if trans in list(transformers['truth'].keys()):
                                    ys[:, col]     = transformers["truth"][trans].inverse_transform(ys[:, col].reshape(-1, 1)).flatten()
                                

                        _, counts    = np.unique(f_event.flatten(), return_counts = True)

                        xs           = np.split(x_long, np.cumsum(counts[: -1]))

                        # print(_, ys[:, 0])
                        
                        # Save in folder
                        with open(osp.join(x_path, data_type + str(i) + ".dat"), "wb") as xy_file:
                            pickle.dump([xs, ys], xy_file)

            if not as_exists:
                # Load data from the xs and generate appropiate adjacency matrices in the a - folder
                
                if self.verbose:
                    print("Making adjacency matrices")

                # def generate_a(filename):
                #     with open(osp.join(x_path, xy_file), "rb") as file:
                #         xs, ys = pickle.load(file)
                #     As = []
                #     for x in xs:
                #         try:
                #             a = A_func(x[:, :3], self.GraphParam)
                #         except:
                #              a = csr_matrix(np.ones(shape = (x.shape[0], x.shape[0])) - np.eye(x.shape[0]))
                #         As.append(a)
                    
                #     with open(osp.join(a_path, xy_file), "wb") as a_file:
                #         pickle.dump(As, a_file)
                    # if self.verbose:
                    #     print(f"{xy_file} produced")

                # with Pool(5) as p:
                #     p.map(generate_a, os.listdir(x_path))
                # count = 0
                # total = len(os.listdir(x_path))
                # for i in range(0, len(os.listdir(x_path)) // 5 + 1, 5):
                #     processes = []
                #     for xy_file in os.listdir(x_path)[i : i + 5]:
                #         p = Process(target = generate_a, args = (xy_file, ))
                #         processes.append(p)
                #         p.start()

                #     for p in processes:
                #         p.join()
                #         count += 1
                #         if self.verbose:
                #             print(f"{count} / {total} done")
                for xy_file in tqdm.tqdm(os.listdir(x_path)):
                    with open(osp.join(x_path, xy_file), "rb") as file:
                        xs, ys = pickle.load(file)
                    As = []

                    for x in xs:
                        try:
                            a = A_func(x[:, :3], self.GraphParam)
                        except:
                             a = csr_matrix(np.ones(shape = (x.shape[0], x.shape[0])) - np.eye(x.shape[0]))
                        As.append(a)
                    
                    with open(osp.join(a_path, xy_file), "wb") as a_file:
                        pickle.dump(As, a_file)


    def generator(self):
        # Define paths
        x_path  = osp.join(file_path, "data", "features", self.Data)
        a_path  = self.path


        file_names = [f for f in os.listdir(x_path) if self.type in f] 

        if self.type == "train":
            n_files = max(self.buffer_size // self.graph_batch, 1)

            np.random.shuffle(file_names)
            file_names = file_names[:n_files]

        x_files = [osp.join(x_path, f) for f in file_names] 
        a_files = [osp.join(a_path, f) for f in file_names] 

        # Define generator for data loading
        def graph_generator():
            
            # Loop over files
            for xy_path, a_path in zip(x_files, a_files):
                
                xy_file = pickle.load(open(xy_path, "rb"))
                # print(xy_file)
                xs, ys = xy_file
                
                As  = pickle.load(open(a_path,  "rb"))

                # Loop over data
                for x, y, a in zip(xs, ys, As):
                    yield Graph(x = x, a = a, y = y)

        return graph_generator()



    def read(self):
        graph_generator = self.generator()
        return [i for i in graph_generator]



