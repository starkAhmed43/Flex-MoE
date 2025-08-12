import os
import ast
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from torch.utils.data import Dataset, DataLoader

class PatchEmbeddings(nn.Module):
    """
    Adapts a 1D feature vector into a sequence of patches for a Transformer.

    This module takes a single, flat feature vector (e.g., a pre-computed embedding)
    and converts it into a sequence of smaller vectors, or "patches." This is a
    necessary preprocessing step to use a Transformer architecture, which expects
    a sequence as input. The naming is inspired by the Vision Transformer (ViT),
    which performs a similar operation on 2D image patches.

    Args:
        feature_size (int): The dimensionality of the input feature vector.
        num_patches (int): The desired number of patches to create in the output sequence.
        embed_dim (int): The target dimensionality for each output patch (the Transformer's hidden size).
    """
    def __init__(self, feature_size, num_patches, embed_dim):
        super().__init__()
        # Calculate the size of each patch. We use ceiling division to ensure
        # the entire feature vector is covered.
        patch_size = math.ceil(feature_size / num_patches)
        
        # Calculate the amount of padding needed to make the feature_size
        # perfectly divisible by the patch_size.
        pad_size = num_patches * patch_size - feature_size
        
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        
        # A linear layer to project each patch from its calculated size
        # to the desired embedding dimension for the Transformer.
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        """
        Defines the forward pass for creating the patch sequence.

        Args:
            x (Tensor): The input tensor of shape (batch_size, feature_size).

        Returns:
            Tensor: The output tensor of shape (batch_size, num_patches, embed_dim).
        """
        # 1. Pad the input tensor on the last dimension to make it evenly divisible.
        #    Input shape: (batch_size, feature_size)
        #    Output shape: (batch_size, feature_size + pad_size)
        padded_x = F.pad(x, (0, self.pad_size))
        
        # 2. Reshape the padded tensor into a sequence of patches.
        #    This is the core "patching" operation.
        #    Output shape: (batch_size, num_patches, patch_size)
        patches = padded_x.view(x.shape[0], self.num_patches, self.patch_size)
        
        # 3. Project each patch to the target embedding dimension.
        #    The linear layer is applied to the last dimension (patch_size).
        #    Output shape: (batch_size, num_patches, embed_dim)
        projected_patches = self.projection(patches)
        
        return projected_patches


class MultiModalDataset(Dataset):
    """
    Custom PyTorch Dataset for multi-modal data.

    This class takes pre-processed data dictionaries and prepares them for loading
    into a model. It handles the indexing and retrieval of individual samples,
    including their multiple data modalities, labels, and metadata.

    Args:
        data_dict (dict): A dictionary where keys are modality names and values are numpy arrays of the data.
                          It also includes 'modality_comb' for combination indices.
        observed_idx (np.ndarray): A boolean array indicating which modalities are present for each sample.
        ids (list): A list of integer indices for the dataset split (e.g., train, valid, test).
        labels (np.ndarray): An array of labels for all samples.
        input_dims (dict): A dictionary mapping modality names to their feature dimensions.
        use_common_ids (bool): Flag to filter dataset to only samples where all modalities are present.
    """
    def __init__(self, data_dict, observed_idx, ids, labels, input_dims, use_common_ids=True):
        self.data_dict = data_dict
        self.mc = np.array(data_dict['modality_comb'])
        self.observed = observed_idx
        self.ids = ids
        self.labels = labels
        self.input_dims = input_dims
        self.use_common_ids = use_common_ids

        # Create a new view of the data containing only the samples for the current split (e.g., training set).
        self.data_new = {modality: data[ids] for modality, data in self.data_dict.items() if 'modality' not in modality}
        self.label_new = self.labels[ids]
        self.mc_new = self.mc[ids]
        self.observed_new = self.observed[ids]

        # Sort samples by the number of available modalities. This can be useful for curriculum learning.
        # Samples with more modalities will appear first.
        self.sorted_ids = sorted(
            np.arange(len(ids)), 
            key=lambda idx: sum([1 for modality in self.data_new if -2 not in self.data_new[modality][idx]]), 
            reverse=True
        )
        # Reorder the data based on the sorted indices.
        self.data_new = {modality: data[self.sorted_ids] for modality, data in self.data_new.items()}
        self.label_new = self.label_new[self.sorted_ids]
        self.mc_new = self.mc_new[self.sorted_ids]
        self.observed_new = self.observed_new[self.sorted_ids]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - sample_data (dict): Data for all modalities for the sample.
                - label (int): The label of the sample.
                - mc (int): The modality combination index.
                - observed (np.ndarray): The observed status for each modality.
        """
        # Fetch data for a single sample.
        sample_data = {modality: data[idx] for modality, data in self.data_new.items()}
        label = self.label_new[idx]
        mc = self.mc_new[idx]
        observed = self.observed_new[idx]

        return sample_data, label, mc, observed

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

def load_and_preprocess_data(args, modality_dict):
    """
    Loads and preprocesses data from pre-split Parquet files.

    This function reads train, validation, and test sets from Parquet files,
    processes embedding columns (which may be stored as strings), and prepares
    all data structures required for creating PyTorch DataLoaders.

    Args:
        args (Namespace): A namespace object containing command-line arguments, such as
                          data paths and configuration options.
        modality_dict (dict): A dictionary mapping modality names (e.g., 'smiles_1d') to
                              their integer index.

    Returns:
        tuple: A comprehensive tuple containing all processed data and metadata:
            - data_dict (dict): Dictionary of numpy arrays for each modality's data.
            - encoder_dict (dict): Dictionary of encoder models for each modality.
            - labels (np.ndarray): Array of all labels.
            - train_idxs, valid_idxs, test_idxs (list): Lists of integer indices for each data split.
            - n_labels (int): The number of unique classes.
            - input_dims (dict): Dictionary of feature dimensions for each modality.
            - transforms (dict): (Empty) Placeholder for data transforms.
            - masks (dict): (Empty) Placeholder for data masks.
            - observed_idx_arr (np.ndarray): Boolean array indicating modality presence.
            - full_modality_index (int): The index representing the combination of all modalities.
    """
    # --- 1. Load Pre-split Data ---
    data_dir = getattr(args, 'data_dir', './data/smiles/')
    id_col = getattr(args, 'id_col', 'ID')
    label_col = getattr(args, 'label_col', 'label')

    try:
        # Load the pre-split datasets from Parquet files.
        train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
        valid_df = pd.read_parquet(os.path.join(data_dir, 'valid.parquet'))
        test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    except FileNotFoundError as e:
        print(f"Error: Make sure `train.parquet`, `valid.parquet`, and `test.parquet` exist in `{data_dir}`.")
        raise e

    # Set the specified ID column as the DataFrame index for easy lookup.
    for df_ in [train_df, valid_df, test_df]:
        df_.set_index(id_col, inplace=True)

    # Combine all data into a single dataframe for unified processing and to create a global index map.
    df = pd.concat([train_df, valid_df, test_df])
    
    labels = df[label_col].values.astype(np.int64)
    n_labels = len(set(labels))

    # Get the original IDs from the pre-split dataframes to preserve the splits.
    train_ids = train_df.index.tolist()
    valid_ids = valid_df.index.tolist()
    test_ids = test_df.index.tolist()

    # --- 2. Initialize Data Structures ---
    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    
    # Create a mapping from the string ID of each sample to a unique integer index.
    id_to_idx = {id: idx for idx, id in enumerate(df.index)}
    common_idx_list = []
    # This array will track which modalities are present for each sample.
    observed_idx_arr = np.zeros((labels.shape[0], args.n_full_modalities), dtype=bool)
    # This list will store the combination of modalities present for each sample as a string (e.g., "124").
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality_char):
        """Helper to update the modality combination string for a given sample."""
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality_char
        else:
            # Append modality character and sort to ensure consistency (e.g., "12" is same as "21").
            modality_combinations[idx] = ''.join(sorted(modality_combinations[idx] + modality_char))

    def parse_embedding(embedding_str):
        """
        Safely parses a string representation of a list/array into a Python list.
        Handles cases where the data is already in list/array format or is missing (NaN).
        """
        try:
            # `ast.literal_eval` is a safe way to evaluate a string containing a Python literal.
            return ast.literal_eval(embedding_str)
        except (ValueError, SyntaxError, TypeError):
            # If parsing fails, check if it's already a list or numpy array.
            if isinstance(embedding_str, (list, np.ndarray)):
                return embedding_str
            # Return None for NaNs or other malformed strings.
            return None 

    # --- 3. Process Each Modality ---
    modality_map = {
        '1': {'col': 'smiles_1d_embedding', 'name': 'smiles_1d'},
        '2': {'col': 'smiles_2d_embedding', 'name': 'smiles_2d'},
        '3': {'col': 'smiles_3d_embedding', 'name': 'smiles_3d'},
        '4': {'col': 'esmc_embedding', 'name': 'esmc'},
        '5': {'col': 'esm3_embedding', 'name': 'esm3'}
    }

    # Loop through each potential modality defined in the map.
    for mod_char, mod_info in modality_map.items():
        # Check if this modality was requested for the current experiment run.
        if mod_char in args.modality:
            mod_col = mod_info['col']
            mod_name = mod_info['name']
            
            if mod_col not in df.columns:
                print(f"Warning: Modality column '{mod_col}' not found in DataFrame. Skipping.")
                continue

            # Parse the embedding column from string to list format.
            raw_embeddings = df[mod_col].apply(parse_embedding)
            # Drop samples where the embedding is missing.
            valid_embeddings = raw_embeddings.dropna()

            if valid_embeddings.empty:
                print(f"Warning: No valid data for modality '{mod_name}'. Skipping.")
                continue

            # Get the dimensionality of the embeddings from the first valid sample.
            embedding_dim = len(valid_embeddings.iloc[0])
            input_dims[mod_name] = embedding_dim
            
            # Stack the valid embeddings into a single numpy array.
            arr = np.vstack(valid_embeddings.values).astype(np.float32)
            
            # Get the integer indices corresponding to the samples that have this modality.
            valid_indices = valid_embeddings.index
            new_idx = np.array(convert_ids_to_index(valid_indices, id_to_idx))
            filtered_idx = new_idx[new_idx != -1]
            
            # Mark this modality as "observed" for these samples.
            observed_idx_arr[filtered_idx, modality_dict[mod_name]] = True
            for idx in filtered_idx:
                update_modality_combinations(idx, mod_char)
            
            # Create the final data matrix for this modality, filling missing samples with -2 as a sentinel value.
            tmp = np.full((len(id_to_idx), embedding_dim), -2.0, dtype=np.float32)
            tmp[filtered_idx] = arr
            
            data_dict[mod_name] = tmp
            common_idx_list.append(set(filtered_idx))
            # Define a simple patch embedding layer for this modality.
            encoder_dict[mod_name] = PatchEmbeddings(embedding_dim, args.num_patches, args.hidden_dim).to(args.device)

    # --- 4. Finalize and Create Splits ---
    combination_to_index = get_modality_combinations(args.modality)
    # The index for the combination with all modalities is expected to be 0.
    full_modality_index = min(combination_to_index.values()) if combination_to_index else -1
    _keys = combination_to_index.keys()
    # Convert the modality combination strings (e.g., "124") to their corresponding integer indices.
    data_dict['modality_comb'] = [combination_to_index.get(comb, -1) for comb in modality_combinations]

    # Convert the original string-based IDs for each split into integer indices.
    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    # If requested, filter all splits to only include samples where all modalities are present.
    if args.use_common_ids and common_idx_list:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs.intersection(train_idxs))
        valid_idxs = list(common_idxs.intersection(valid_idxs))
        test_idxs = list(common_idxs.intersection(test_idxs))

    def all_modalities_missing(idx):
        """Check if a sample is missing all of its modality data."""
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict if modality != 'modality_comb')

    # Clean the training set of any samples that have no modalities at all.
    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    # Return empty dicts for transforms and masks as they are not used for pre-computed embeddings.
    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, {}, {}, observed_idx_arr, full_modality_index

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

def create_loaders(data_dict, observed_idx, labels, train_ids, valid_ids, test_ids, batch_size, num_workers, pin_memory, input_dims, use_common_ids=True):
    """
    Creates PyTorch DataLoader instances for training, validation, and testing.

    Args:
        (See MultiModalDataset and load_and_preprocess_data for argument descriptions)
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.

    Returns:
        tuple: A tuple containing the DataLoader objects:
            - train_loader, train_loader_shuffle, val_loader, test_loader
    """
    
    # Instantiate the custom dataset for each split.
    train_dataset = MultiModalDataset(data_dict, observed_idx, train_ids, labels, input_dims, use_common_ids)
    valid_dataset = MultiModalDataset(data_dict, observed_idx, valid_ids, labels, input_dims, use_common_ids)
    test_dataset = MultiModalDataset(data_dict, observed_idx, test_ids, labels, input_dims, use_common_ids)

    # Create DataLoader instances for each dataset.
    # A shuffled loader is provided for training epochs.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    train_loader_shuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, train_loader_shuffle, val_loader, test_loader

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