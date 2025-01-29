import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset, TensorDataset


def data_split(dataset, val_portion=0.2, test_portion=0.2, random_state=0):

    rng = np.random.default_rng(random_state)
    inds = np.arange(len(dataset))
    rng.shuffle(inds)

    # Assign indices to splits.
    n_val = int(val_portion * len(dataset))
    n_test = int(test_portion * len(dataset))
    test_inds = inds[:n_test]
    val_inds = inds[n_test:(n_test + n_val)]
    train_inds = inds[(n_test + n_val):]

    # Create split datasets.
    test_dataset = Subset(dataset, test_inds)
    val_dataset = Subset(dataset, val_inds)
    train_dataset = Subset(dataset, train_inds)
    return train_dataset, val_dataset, test_dataset


def subset_to_dataframe(subset):

    dataset = subset.dataset
    indices = subset.indices

    data = [dataset.iloc[i] for i in indices]

    return pd.DataFrame(data)







