import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader

import scipy.sparse as sp
from os import getenv
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split


class DataProvider(object):
    def __init__(self):

        self.df_classes = pd.read_csv(getenv("TXS_CLASSES_NAME"))
        self.df_edges = pd.read_csv(getenv("TXS_EDGE_LIST_NAME"))
        self.df_features = pd.read_csv(getenv("TXS_FEATURES_NAME"), header=None)

    def get_dataset(self):

        self.df_classes["class"] = self.df_classes["class"].map(
            {"unknown": 2, "1": 1, "2": 0}
        )

        df_merge = self.df_features.merge(
            self.df_classes, how="left", right_on="txId", left_on=0
        )
        df_merge = df_merge.sort_values(0).reset_index(drop=True)

        nodes = df_merge[0].values

        map_id = {j: i for i, j in enumerate(nodes)}  # mapping nodes to indexes

        # Create edge df that has transID mapped to nodeIDs
        edges = self.df_edges.copy()
        edges.txId1 = edges.txId1.map(
            map_id
        )  # get nodes idx1 from edges list and filtered data
        edges.txId2 = edges.txId2.map(map_id)

        edges = edges.astype(int)

        edge_index = np.array(edges.values).T  # convert into an array
        edge_index = torch.tensor(
            edge_index, dtype=torch.long
        ).contiguous()  # create a tensor

        weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)

        labels = df_merge["class"].values

        node_features = df_merge.drop(["txId"], axis=1).copy()
        # node_features[0] = node_features[0].map(map_id) # Convert transaction ID to node ID \
        print("unique=", node_features["class"].unique())

        # Retain known vs unknown IDs
        classified_idx = (
            node_features["class"].loc[node_features["class"] != 2].index
        )  # filter on known labels
        unclassified_idx = node_features["class"].loc[node_features["class"] == 2].index

        classified_illicit_idx = (
            node_features["class"].loc[node_features["class"] == 1].index
        )  # filter on illicit labels
        classified_licit_idx = (
            node_features["class"].loc[node_features["class"] == 0].index
        )  # filter on licit labels

        # Drop unwanted columns, 0 = transID, 1=time period, class = labels
        node_features = node_features.drop(columns=[0, 1, "class"])

        # Convert to tensor
        node_features_t = torch.tensor(
            np.array(node_features.values, dtype=np.double), dtype=torch.double
        )  # drop unused columns

        train_idx, valid_test_idx = train_test_split(
            classified_idx.values, test_size=0.3
        )
        valid_idx, test_idx = train_test_split(valid_test_idx, test_size=0.5)

        data_train = Data(
            x=node_features_t,
            edge_index=edge_index,
            edge_attr=weights,
            y=torch.tensor(labels, dtype=torch.double),
        )
        # Add in the train and valid idx
        data_train.train_idx = train_idx
        data_train.valid_idx = valid_idx
        data_train.test_idx = test_idx
        data_train.unclassified_idx = unclassified_idx

        return data_train
