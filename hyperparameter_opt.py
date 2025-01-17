import pandas as pd
import numpy as np
import torch
import scripts
from functools import lru_cache
import torchmetrics
from torch import nn
import optuna

@lru_cache(maxsize = None)
def get_data(n_fold = 0, fp_radius = 2):
    smile_dict = pd.read_csv("data/smiles.csv", index_col=0)
    fp = scripts.FingerprintFeaturizer(R = fp_radius)
    drug_dict = fp(smile_dict.iloc[:, 1], smile_dict.iloc[:, 0])
    driver_genes = pd.read_csv("data/driver_genes.csv").loc[:, "symbol"].dropna()
    rnaseq = pd.read_csv("data/rnaseq_normcount.csv", index_col=0)
    driver_columns = rnaseq.columns.isin(driver_genes)
    filtered_rna = rnaseq.loc[:, driver_columns]
    tensor_exp = torch.Tensor(filtered_rna.to_numpy())
    cell_dict = {cell: tensor_exp[i] for i, cell in enumerate(filtered_rna.index.to_numpy())}
    data = pd.read_csv("data/GDSC12.csv", index_col=0)
    # default, remove data where lines or drugs are missing:
    data = data.query("SANGER_MODEL_ID in @cell_dict.keys() & DRUG_ID in @drug_dict.keys()")
    unique_cell_lines = data.loc[:, "SANGER_MODEL_ID"].unique()
    np.random.seed(420) # for comparibility, don't change it!
    np.random.shuffle(unique_cell_lines)
    folds = np.array_split(unique_cell_lines, 10)
    test_lines = folds[0]
    train_idxs = list(range(10))
    train_idxs.remove(n_fold)
    np.random.seed(420)
    validation_idx = np.random.choice(train_idxs)
    train_idxs.remove(validation_idx)
    train_lines = np.concatenate([folds[idx] for idx in train_idxs])
    validation_lines = folds[validation_idx]
    test_lines = folds[n_fold]
    train_data = data.query("SANGER_MODEL_ID in @train_lines")
    validation_data = data.query("SANGER_MODEL_ID in @validation_lines")
    test_data = data.query("SANGER_MODEL_ID in @test_lines")
    return (scripts.OmicsDataset_drugwise(cell_dict, drug_dict, train_data),
    scripts.OmicsDataset_drugwise(cell_dict, drug_dict, validation_data),
    scripts.OmicsDataset_drugwise(cell_dict, drug_dict, test_data))

config = {"features" : {"fp_radius":2},
          "optimizer": {"batch_size": 3,
                        "clip_norm":19,
                        "learning_rate":0.0004592646200179472,
                        "stopping_patience":15},
          "model":{"embed_dim":485,
                 "hidden_dim":696,
                 "dropout":0.48541242824674574,
                 "n_layers": 4,
                 "norm": "batchnorm"},
         "env": {"fold": 0,
                 "device":"cpu",
                 "max_epochs": 100,
                 "search_hyperparameters":True}}

train_dataset, validation_dataset, test_dataset = get_data(n_fold = config["env"]["fold"],
                                                           fp_radius = config["features"]["fp_radius"])


def collate_fn_custom(batch):
    omics = []
    drugs = []
    targets = []
    cell_ids = []
    drug_ids = []
    labels = [] 

    for o, d, t, cid, did, r in batch:
        omics.append(o)  # Add each omics tensor
        drugs.append(d)  # Add each drugs tensor
        targets.append(t)  # Add each target tensor
        cell_ids.append(cid)  # Add each cell ID tensor
        drug_ids.append(did)  # Add each drug ID tensor
        labels.append(r)  # Add each label tensor

    # Concatenate along the first dimension for all tensors
    omics = torch.cat(omics, dim=0)
    drugs = torch.cat(drugs, dim=0)
    targets = torch.cat(targets, dim=0)
    cell_ids = torch.cat(cell_ids, dim=0)
    drug_ids = torch.cat(drug_ids, dim=0)
    labels = torch.cat(labels, dim=0)

    return omics, drugs, targets, cell_ids, drug_ids, labels


def train_model_optuna(trial, config):
    def pruning_callback(epoch, train_r):
        trial.report(train_r, step = epoch)
        if np.isnan(train_r):
            raise optuna.TrialPruned()
        if trial.should_prune():
            raise optuna.TrialPruned()
    config["model"] = {"embed_dim": trial.suggest_int("embed_dim", 64, 512),
                    "hidden_dim": trial.suggest_int("hidden_dim", 64, 2048),
                    "n_layers": trial.suggest_int("n_layers", 1, 6),
                    "norm": trial.suggest_categorical("norm", ["batchnorm", "layernorm", None]),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.5)}
    config["optimizer"] = { "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
                            "clip_norm": trial.suggest_int("clip_norm", 0.1, 20),
                            "batch_size": trial.suggest_int("batch_size", 2, 10),
                            "stopping_patience":10}
    try:
        R, model = scripts.train_model(config,
                                       train_dataset,
                                       validation_dataset,
                                       use_momentum=True,
                                       callback_epoch = pruning_callback,
                                       collate_fn=collate_fn_custom)
        
        return R
    except Exception as e:
        print(e)
        return 0

if config["env"]["search_hyperparameters"]:
    study_name = f"baseline_model"
    storage_name = "sqlite:///studies/{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                direction='maximize',
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=30,
                                                               n_warmup_steps=5,
                                                               interval_steps=5))
    objective = lambda x: train_model_optuna(x, config)
    study.optimize(objective, n_trials=40)
    best_config = study.best_params
    print(best_config)
    config["model"]["embed_dim"] = best_config["embed_dim"]
    config["model"]["hidden_dim"] = best_config["hidden_dim"]
    config["model"]["n_layers"] = best_config["n_layers"]
    config["model"]["norm"] = best_config["norm"]
    config["model"]["dropout"] = best_config["dropout"]
    config["optimizer"]["learning_rate"] = best_config["learning_rate"]
    config["optimizer"]["clip_norm"] = best_config["clip_norm"]
    config["optimizer"]["batch_size"] = best_config["batch_size"]
