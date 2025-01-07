from torch.utils.data import Dataset
from torch import Tensor
import torch
import numpy as np



class OmicsDataset(Dataset):
    def __init__(self, omic_dict, drug_dict, data):
        self.omic_dict = omic_dict
        self.drug_dict = drug_dict
        self.cell_mapped_ids = {key:i for i, key in enumerate(self.omic_dict.keys())}
        self.drug_mapped_ids = {key:i for i, key in enumerate(self.drug_dict.keys())}
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        instance = self.data.iloc[idx]
        cell_id = instance.iloc[0]
        drug_id = instance.iloc[1]
        target = instance.iloc[2]
        return (self.omic_dict[cell_id],
                self.drug_dict[drug_id],
                Tensor([target]),
                Tensor([self.cell_mapped_ids[cell_id]]),
                Tensor([self.drug_mapped_ids[drug_id]]))
    

class OmicsDataset_drugwise(Dataset):
    def __init__(self, omic_dict, drug_dict, data):
        self.omic_dict = omic_dict
        self.drug_dict = drug_dict
        self.data = data 

        self.grouped_data = self.data.groupby("DRUG_ID")  # Group data by drug ID
        self.drug_ids = list(self.grouped_data.groups.keys())  # Unique drug IDs

        self.drug_mapped_ids = {key: i for i, key in enumerate(self.drug_ids)}
        self.cell_mapped_ids = {key: i for i, key in enumerate(self.omic_dict.keys())}

    def __len__(self):
        return len(self.drug_ids)  # Number of unique drugs
    
    def pad_tensor(self, vec, pad, dim):
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

    def __getitem__(self, idx):

        drug_id = self.drug_ids[idx] # Get the drug ID corresponding to the index
        drug_group = self.grouped_data.get_group(drug_id) # Get the rows corresponding to this drug
        
        cell_ids = drug_group["SANGER_MODEL_ID"].values
        labels = drug_group["label"].values  # Use labels as-is
        target = drug_group["LN_IC50"].values 
        
        # Convert to tensors
        drug_features = Tensor(self.drug_dict[drug_id])  # Drug features
         # Repeat drug features for all cell lines
        # cell_features = Tensor([self.omic_dict[cell_id] for cell_id in cell_ids])  # All cell features
        cell_features = torch.stack([self.omic_dict[cell_id].clone().detach().float() for cell_id in cell_ids])  # All cell features


        #xcell_features = self.pad_tensor(cell_features, 1000, 0)

        labels = Tensor(labels)  # Labels as a tensors
        #labels = self.pad_tensor(labels, 1000, 0)

        target = Tensor(target)
        #target = self.pad_tensor(target, 1000, 0)

        cell_num = len(labels)
        drug_features = drug_features.repeat(cell_num, 1) 
        drug_index = Tensor([self.drug_mapped_ids[drug_id]])  # Drug index
        drug_index = drug_index.repeat(cell_num)
        cell_indices = Tensor([self.cell_mapped_ids[cell_id] for cell_id in cell_ids])  # Cell indices
        #cell_indices = self.pad_tensor(cell_indices, 1000, 0)

        return (cell_features , 
                drug_features, 
                target, 
                drug_index, 
                cell_indices,
                labels )
    


    
import rdkit
from rdkit.Chem import AllChem
class FingerprintFeaturizer():
    def __init__(self,
                 fingerprint = "morgan",
                 R=2, 
                 fp_kwargs = {},
                 transform = Tensor):
        """
        Get a fingerprint from a list of molecules.
        Available fingerprints: MACCS, morgan, topological_torsion
        R is only used for morgan fingerprint.
        fp_kwards passes the arguments to the rdkit fingerprint functions:
        GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint, GetTopologicalTorsionFingerprint
        """
        self.R = R
        self.fp_kwargs = fp_kwargs
        self.fingerprint = fingerprint
        if fingerprint == "morgan":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(x, self.R, **fp_kwargs)
        elif fingerprint == "MACCS":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint(x, **fp_kwargs)
        elif fingerprint == "topological_torsion":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetTopologicalTorsionFingerprint(x, **fp_kwargs)
        self.transform = transform
    def __call__(self, smiles_list, drugs = None):
        drug_dict = {}
        if drugs is None:
            drugs = np.arange(len(smiles_list))
        for i in range(len(smiles_list)):
            try:
                smiles = smiles_list[i]
                molecule = AllChem.MolFromSmiles(smiles)
                feature_list = self.f(molecule)
                f = np.array(feature_list)
                if self.transform is not None:
                    f = self.transform(f)
                drug_dict[drugs[i]] = f
            except:
                drug_dict[drugs[i]] = None
        return drug_dict
    def __str__(self):
        """
        returns a description of the featurization
        """
        return f"{self.fingerprint}Fingerprint_R{self.R}_{str(self.fp_kwargs)}"
