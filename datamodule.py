import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from itertools import combinations
from torch.utils.data import Dataset, DataLoader

def convert_ids_to_index(ids, index_map):
    """
    Converts a list of string-based sample IDs to their corresponding integer indices.

    Args:
        ids (list): A list of string IDs.
        index_map (dict): A mapping from string ID to integer index.

    Returns:
        list: A list of integer indices. Returns -1 for IDs not found in the map.
    """
    return [index_map.get(id, -1) for id in ids]

def collate_fn(batch):
    """
    Custom collate function to handle batches of multi-modal data.

    This function takes a list of samples (each a tuple from MultiModalDataset)
    and organizes them into batches of tensors.

    Args:
        batch (list): A list of samples from the MultiModalDataset.

    Returns:
        tuple: A tuple containing batched data:
            - collated_data (dict): A dictionary of tensors, one for each modality.
            - labels (Tensor): A tensor of labels.
            - mcs (Tensor): A tensor of modality combination indices.
            - observeds (Tensor): A boolean tensor of observed statuses.
    """
    # Unzip the batch into separate lists for data, labels, etc.
    data, labels, mcs, observeds = zip(*batch)
    
    if not data:
        return {}, torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Get modality names from the first sample in the batch.
    modalities = data[0].keys()
    
    # Stack the data for each modality from all samples in the batch into a single tensor.
    collated_data = {modality: torch.tensor(np.stack([d[modality] for d in data]), dtype=torch.float32) for modality in modalities}
    
    # Convert lists of labels, mcs, and observeds into tensors.
    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds), dtype=torch.bool)
    
    return collated_data, labels, mcs, observeds

def get_modality_combinations(modalities):
    """
    Generates a mapping from modality combinations (e.g., '123') to integer indices.

    The combination with the most modalities is mapped to 0, the next to 1, and so on.
    This is useful for some multi-modal learning strategies.

    Args:
        modalities (str): A string containing the characters for each modality (e.g., "12345").

    Returns:
        dict: A dictionary mapping sorted combination strings to integer indices.
    """
    all_combinations = []
    # Iterate from the full set of modalities down to single modalities.
    for i in range(len(modalities), 0, -1):
        comb = list(combinations(modalities, i))
        all_combinations.extend(comb)
    
    # Create the mapping from the sorted string version of the combination to its index.
    combination_to_index = {''.join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)}
    return combination_to_index

class FlexMoEDataset(Dataset):
    """
    Bridge Dataset: Converts a DataFrame with your data structure into the format expected by MultiModalDataset in data.py.
    """
    def __init__(self, dataframe, modalities, label_col, modality_dict):
        """
        Initialize the FlexMoEDataset.

        Args:
            dataframe (pd.DataFrame): Your data, already cleaned and imputed.
            modalities (list): List of modality column names (e.g., ['smiles_1d_embedding', ...]).
            label_col (str): Name of the label column.
            modality_dict (dict): Mapping from model's modality names to indices (e.g., {'smiles_1d': 0, ...}).
        """
        self.df = dataframe.reset_index(drop=True)
        self.modalities = modalities
        self.label_col = label_col
        self.modality_dict = modality_dict

        # Build observed array: shape (num_samples, num_modalities)
        self.observed = np.zeros((len(self.df), len(modality_dict)), dtype=bool)
        for i, mod in enumerate(modality_dict):
            col = mod  # Now col is the full column name, e.g., 'smiles_1d_embedding'
            if col in self.df.columns:
                self.observed[:, i] = self.df[col].apply(lambda x: x is not None and not (isinstance(x, float) and np.isnan(x)))
            else:
                self.observed[:, i] = False

        # Build modality combination index (as in data.py)
        self.modality_combinations = []
        for obs in self.observed:
            present = [str(i+1) for i, o in enumerate(obs) if o]
            self.modality_combinations.append(''.join(present) if present else '-1')

        # Map combination string to index (as in get_modality_combinations)
        all_mods = ''.join([str(i+1) for i in range(len(modality_dict))])
        self.comb_to_idx = get_modality_combinations(all_mods)
        self.mc = np.array([self.comb_to_idx.get(comb, -1) for comb in self.modality_combinations])

        # Labels
        self.labels = self.df[label_col].values

        # Sort samples by number of available modalities (descending)
        num_present = self.observed.sum(axis=1)
        sorted_indices = np.argsort(-num_present)  # Descending order

        # Reorder everything
        self.df = self.df.iloc[sorted_indices].reset_index(drop=True)
        self.observed = self.observed[sorted_indices]
        self.modality_combinations = [self.modality_combinations[i] for i in sorted_indices]
        self.mc = self.mc[sorted_indices]
        self.labels = self.labels[sorted_indices]

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (sample_data, label, mc, observed)
                sample_data (dict): Dictionary of modality arrays.
                label: Label for the sample.
                mc: Modality combination index.
                observed: Boolean array indicating observed modalities.
        """
        # Build sample_data dict as in MultiModalDataset
        sample_data = {}
        for mod in self.modality_dict:
            col = mod  # Use the full column name directly
            if col in self.df.columns:
                arr = self.df.iloc[idx][col]
                # If missing, fill with -2 (as in data.py)
                if arr is None or (isinstance(arr, float) and np.isnan(arr)):
                    arr = np.full_like(self.df.iloc[0][col], -2.0)
                sample_data[mod] = np.array(arr, dtype=np.float32)
            else:
                # If modality not present, fill with -2
                arr = np.full_like(self.df.iloc[0][self.modalities[0]], -2.0)
                sample_data[mod] = np.array(arr, dtype=np.float32)
        label = self.labels[idx]
        mc = self.mc[idx]
        observed = self.observed[idx]
        return sample_data, label, mc, observed

class FlexMoEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        train_file="train.parquet",
        val_file="val.parquet",
        test_file="test.parquet",
        modalities=[
            'smiles_1d_embedding', 'smiles_2d_embedding', 'smiles_3d_embedding',
            'smiles_clamp_embedding', 'ecfp', 'atom_pair_fp',
            'esmc_embedding', 'esm3_embedding'
        ],
        label_col="log10_value",
        batch_size=1024,
        num_workers=16,
        pin_memory=True,
        modality_dict=None,
    ):
        """
        Initialize the FlexMoEDataModule.

        Args:
            data_dir (str): Directory containing the data files.
            train_file (str): Filename for the training data.
            val_file (str): Filename for the validation data.
            test_file (str): Filename for the test data.
            modalities (list): List of modality column names.
            label_col (str): Name of the label column.
            batch_size (int): Batch size for data loaders.
            num_workers (int): Number of worker processes for data loading.
            pin_memory (bool): Whether to use pinned memory in data loaders.
            modality_dict (dict): Mapping from modality names to indices.
        """
        super().__init__()
        self.data_dir = data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.modalities = modalities
        self.label_col = label_col
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.modality_dict = modality_dict or {mod: i for i, mod in enumerate(self.modalities)}

    def setup(self, stage=None):
        """
        Set up datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage to set up. Not used, included for Lightning compatibility.
        """
        # Load DataFrames
        train_df = pd.read_parquet(f"{self.data_dir}/{self.train_file}")
        val_df = pd.read_parquet(f"{self.data_dir}/{self.val_file}")
        test_df = pd.read_parquet(f"{self.data_dir}/{self.test_file}")

        # Create datasets
        self.train_dataset = FlexMoEDataset(
            train_df, self.modalities, self.label_col, self.modality_dict
        )
        self.val_dataset = FlexMoEDataset(
            val_df, self.modalities, self.label_col, self.modality_dict
        )
        self.test_dataset = FlexMoEDataset(
            test_df, self.modalities, self.label_col, self.modality_dict
        )

    def train_dataloader(self):
        """
        Get the training data loader.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """
        Get the validation data loader.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """
        Get the test data loader.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_input_dims(self):
        """
        Get the input dimension of each modality in the dataset.

        Returns:
            list: List of input dimensions for each modality.
        """
        # Returns a list of input dims for each modality (for model construction)
        if not hasattr(self, "train_dataset") or self.train_dataset is None:
            self.setup()
        
        input_dims = []
        for mod in self.modalities:
            first_valid = next(
                (x for x in self.train_dataset.df[mod] if x is not None and not (isinstance(x, float) and np.isnan(x))),
                None
            )
            if first_valid is None:
                raise ValueError(f"No valid embedding found for modality '{mod}' in train_df!")
            input_dims.append(np.array(first_valid).shape[-1])
        return input_dims

if __name__ == "__main__":
    datamodule = FlexMoEDataModule(data_dir="/home/adhil/MLRxnDB/data/datasets/kinetic_dataset/S4")
    print(f"Input dimensions: {datamodule.get_input_dims()}")