import torch
import numpy as np
import os


def mnist(train=True):
    corrupt_path = "../../../data/corruptmnist"

    extension = "train" if train else "test"
    files = [np.load(os.path.join(corrupt_path, x) )
            for x in os.listdir(corrupt_path) if extension in x]
    
    images = torch.from_numpy(
        np.concatenate([x["images"] for x in files], axis=0)
    )
    labels = torch.from_numpy(
        np.concatenate([x["labels"] for x in files], axis=0)
    )

    return images, labels



