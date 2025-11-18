"""
Evaluating the performance of the model

data directory should contain the following:

- wandb logged data, extracting loss from it
- config file .yaml
- model checkpoints

"""

import torch as t
import pandas as pd
import yaml
import os

def load_data(data_dir: str):
    """
    Load the log, statistics, and metadata as well
    from timestamped directory
    """
    metadata = yaml.load(open(os.path.join(data_dir, "config.yaml")))
    data = pd.read_parquet(os.path.join(data_dir, "training_metrics.parquet"))
    
    return metadata, data


