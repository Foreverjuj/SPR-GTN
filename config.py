import ml_collections
import copy
import os

CONFIG = ml_collections.ConfigDict({
    "debug": False,
    "max_epochs": 15,
    "batch_size": 32,

    "device": 'cuda:7',
    "optimizer":{
        "lr": 1e-4,
        "betas": (0.95, 0.95),
        "eps": 1e-8,
    },
    "scheduler":{
        "step_size": 300,
        "gamma": 0.75,
    },

    "model_save_path": "model_weight/model_",
    "loss_save_path": "model_weight/loss_",
    "test_result_path": "model_weight/test_",
})

def get_config():
    config = copy.deepcopy(CONFIG)
    return config


