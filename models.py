import numpy as np
from itertools import combinations

import torch
import torch.nn as nn

# Import the custom MoE modules from the other file.
from moe_module import *


class FlexMoE(nn.Module):
    """
    The main model class that orchestrates the entire Flex-MoE architecture.

    This class constructs a multi-layer Transformer-based network that can handle
    multiple input modalities. It dynamically creates expert indices for different
    combinations of available modalities and is responsible for passing this routing
    information to the underlying MoE layers. It also provides a method to collect
    the crucial auxiliary gate loss required for stable MoE training.
    """
    def __init__(self, num_modalities, full_modality_index, num_patches, hidden_dim, output_dim, num_layers, num_layers_pred, num_experts, num_routers, top_k, num_heads=2, dropout=0.5):
        """
        Initializes the FlexMoE model.

        Args:
            num_modalities (int): The total number of modalities in the dataset.
            full_modality_index (int): The expert index reserved for the G-Router (when all modalities are present).
            num_patches (int): The number of patches or tokens for each modality's input.
            hidden_dim (int): The main hidden dimension used throughout the Transformer layers.
            output_dim (int): The final output dimension for the prediction head.
            num_layers (int): The number of Transformer encoder layers.
            num_layers_pred (int): The number of layers in the final MLP prediction head.
            num_experts (int): The number of experts in each MoE layer. Should be >= 2^num_modalities - 1.
            num_routers (int): The number of routers (gating networks) in the MoE layers.
            top_k (int): The number of experts to route each token to.
            num_heads (int, optional): The number of attention heads. Defaults to 2.
            dropout (float, optional): The dropout rate. Defaults to 0.5.
        """
        super(FlexMoE, self).__init__()
        layers = []
        _sparse = True # Start with a sparse (MoE) layer.

        # Append the first Transformer layer.
        layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=_sparse, full_modality_index=full_modality_index, top_k=top_k))
        
        # Create the remaining layers, alternating between sparse (MoE) and dense MLP layers.
        for j in range(num_layers - 1):
            _sparse = not _sparse
            layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=_sparse, full_modality_index=full_modality_index, top_k=top_k))
        
        # Append the final prediction head, which is a standard MLP.
        layers.append(MLP(hidden_dim*num_modalities, hidden_dim, output_dim, num_layers_pred, activation=nn.ReLU(), dropout=0.5))
        
        # Register all layers as a sequential network.
        self.network = nn.Sequential(*layers)
        # Create learnable positional embeddings for the concatenated input tokens from all modalities.
        self.pos_embed = nn.Parameter(torch.zeros(1, np.sum([num_patches]*num_modalities), hidden_dim))
        # Create the mapping from each possible modality combination to a unique expert index.
        self.combination_to_index = self._create_combination_index(num_modalities)

    def forward(self, *inputs, expert_indices=None, is_full_modality=None):
        """
        Defines the forward pass of the model.

        Args:
            *inputs (torch.Tensor): A variable number of tensors, one for each input modality.
            expert_indices (torch.Tensor, optional): Pre-computed expert indices for the S-Router. Defaults to None.
            is_full_modality (bool, optional): Flag indicating if the batch is for the G-Router. Defaults to None.

        Returns:
            torch.Tensor: The final prediction logits.
        """
        # Get the number of tokens for each modality to split them back later.
        chunk_size = [input.shape[1] for input in inputs]
        # Concatenate all modality inputs along the token dimension.
        x = torch.cat(inputs, dim=1)
        
        # Add positional embeddings if they exist.
        if self.pos_embed != None:
            x += self.pos_embed

        # Split the concatenated tensor back into a list of tensors, one for each modality.
        x = torch.split(x, chunk_size, dim=1)

        # Pass the data through the Transformer encoder layers.
        for i in range(len(self.network) - 1):
            # If expert indices are provided, set them in the current layer.
            if expert_indices is not None and hasattr(self.network[i], 'set_expert_index'):
                self.network[i].set_expert_index(expert_indices)
            x = self.network[i](x)
        
        # After the Transformer layers, perform mean pooling over the token dimension for each modality.
        x = [item.mean(dim=1) for item in x]
        # Concatenate the pooled features from all modalities.
        x = torch.cat(x, dim=1)
        # Pass the final feature vector through the MLP prediction head.
        x = self.network[-1](x)
        return x

    def gate_loss(self):
        """
        Collects the auxiliary load-balancing losses from all MoE gates in the network.

        This loss is crucial for stable training of Mixture-of-Experts models as it
        encourages the gates to distribute tokens evenly across all experts.

        Returns:
            torch.Tensor: The sum of all gate losses.
        """
        g_loss = []
        # Iterate over all named modules in the network.
        for mn, mm in self.named_modules():
            # Check if a module has the 'all_gates' attribute, which identifies it as an MoE layer.
            if hasattr(mm, 'all_gates'):
                # Iterate through each gate in the MoE layer.
                for i in range(len(mm.all_gates)):
                    # Retrieve the loss calculated during the forward pass of the gate.
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is None:
                        print(f"[WARN] The gate loss if {mn}, modality: {i} is emtpy, check weather call <get_loss> twice.")
                    else:
                        g_loss.append(i_loss)
        
        # Return the sum of all collected gate losses.
        return sum(g_loss)

    def _create_combination_index(self, num_modalities):
        """
        Creates a dictionary mapping every possible combination of modalities to a unique integer index.

        This is the core mechanism for the Specialized Router (S-Router), allowing the model
        to assign a specific expert to each unique pattern of present/missing modalities.

        Args:
            num_modalities (int): The total number of modalities.

        Returns:
            dict: A dictionary mapping from a sorted tuple of modality indices to an expert index.
        """
        combinations_list = []
        # Generate all non-empty combinations of modality indices.
        for r in range(1, num_modalities + 1):
            combinations_list.extend(combinations(range(num_modalities), r))
        
        # Assign a unique index (starting from 0) to each combination.
        combination_to_index = {tuple(sorted(comb)): idx for idx, comb in enumerate(combinations_list)}
        return combination_to_index

    def assign_expert(self, combination):
        """
        Looks up the expert index for a given combination of present modalities.

        Args:
            combination (tuple or list): A collection of indices for the modalities that are present.

        Returns:
            int: The unique expert index corresponding to the combination.
        """
        index = self.combination_to_index.get(tuple(sorted(combination)))
        return index

    def set_full_modality(self, is_full_modality):
        """
        Propagates a flag to all layers to indicate if the current batch contains only
        samples with all modalities present. This allows the gates to switch their loss
        calculation logic between the G-Router and S-Router.

        Args:
            is_full_modality (bool): True if the batch is for the G-Router.
        """
        for layer in self.network:
            if hasattr(layer, 'set_full_modality'):
                layer.set_full_modality(is_full_modality)

class MLP(nn.Module):
    """
    A standard Multi-Layer Perceptron (MLP) module.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        """
        Initializes the MLP.

        Args:
            input_dim (int): Dimension of the input layer.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output layer.
            num_layers (int): The total number of linear layers.
            activation (nn.Module, optional): The activation function. Defaults to nn.ReLU().
            dropout (float, optional): The dropout rate. Defaults to 0.5.
        """
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)

        if num_layers == 1:
            # If only one layer, it's a direct mapping from input to output.
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # First layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            # Intermediate hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            # Final layer
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """The forward pass for the MLP."""
        return self.network(x)


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, with an optional sparse MoE feed-forward network.
    """
    def __init__(self, 
                num_experts,
                num_routers,
                d_model, 
                num_head, 
                dropout=0.1, 
                activation=nn.GELU, 
                hidden_times=2, 
                mlp_sparse = False, 
                self_attn = True,
                full_modality_index=4,
                top_k=2,
                **kwargs) -> None:
        """
        Initializes the TransformerEncoderLayer.

        Args:
            num_experts (int): Number of experts for the MoE layer.
            num_routers (int): Number of routers for the MoE layer.
            d_model (int): The main hidden dimension.
            num_head (int): The number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            activation (nn.Module, optional): Activation function. Defaults to nn.GELU.
            hidden_times (int, optional): Multiplier for the MLP hidden dimension. Defaults to 2.
            mlp_sparse (bool, optional): If True, use a sparse MoE layer for the FFN. Otherwise, use a dense MLP. Defaults to False.
            self_attn (bool, optional): If True, perform self-attention over all modalities. Otherwise, performs a form of cross-attention. Defaults to True.
            full_modality_index (int, optional): The expert index for the G-Router. Defaults to 4.
            top_k (int, optional): The number of experts to route to. Defaults to 2.
        """
        super(TransformerEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.attn = Attention(
            d_model, num_heads=num_head, qkv_bias=False, attn_drop=dropout, proj_drop=dropout)
        
        self.mlp_sparse = mlp_sparse
        self.self_attn = self_attn
        self.expert_index = None
        self.full_modality_index = full_modality_index

        # Conditionally define the feed-forward network (MLP).
        if self.mlp_sparse:
            # Use the sparse Mixture-of-Experts MLP.
            self.mlp = FMoETransformerMLP(num_expert=num_experts, n_router=num_routers, d_model=d_model, d_hidden=d_model * hidden_times, activation=nn.GELU(), top_k=top_k, **kwargs)
        else:
            # Use a standard dense MLP.
            self.mlp = MLP(input_dim=d_model, hidden_dim=d_model * hidden_times, output_dim=d_model, num_layers=2, activation=nn.GELU(), dropout=dropout)

    def forward(self, x, attn_mask = None):
        """The forward pass for the Transformer layer."""
        if self.self_attn:
            # --- Self-Attention Path ---
            # All modalities attend to all other modalities.
            chunk_size = [item.shape[1] for item in x]
            x = self.norm1(torch.cat(x, dim=1))
            kv = x # Keys and values are the same as the query for self-attention.
            x = self.attn(x, kv, attn_mask)
            x = x + self.dropout1(x)
            x = torch.split(x, chunk_size, dim=1)
            x = [item for item in x]

            # Apply the second residual connection with the MLP block.
            if self.mlp_sparse:
                # Pass expert indices to the sparse MLP.
                x = [x[i] + self.dropout2(self.mlp(self.norm2(x[i]), self.expert_index)) for i in range(len(chunk_size))]
            else:
                x = [x[i] + self.dropout2(self.mlp(self.norm2(x[i]))) for i in range(len(chunk_size))]
        else:
            # --- Cross-Attention Path ---
            # Each modality attends to all other modalities.
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                x[i] = self.attn(x[i], other_m, attn_mask)
            
            # First residual connection.
            x = [x[i] + self.dropout1(x[i]) for i in range(len(chunk_size))]
            
            # Second residual connection with the MLP block.
            if self.mlp_sparse:
                x = [x[i] + self.dropout2(self.mlp(self.norm2(x[i]), self.expert_index)) for i in range(len(chunk_size))]
            else:
                x = [x[i] + self.dropout2(self.mlp(self.norm2(x[i]))) for i in range(len(chunk_size))]

        return x

    def set_expert_index(self, expert_index):
        """Sets the expert indices to be used by the sparse MLP."""
        self.expert_index = expert_index

    def set_full_modality(self, is_full_modality):
        """Propagates the full modality flag to the sparse MLP."""
        if hasattr(self.mlp, 'set_full_modality'):
            self.mlp.set_full_modality(is_full_modality)

class Attention(nn.Module):
    """
    A standard multi-head self-attention module.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Initializes the Attention module.

        Args:
            dim (int): Input and output dimension.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to False.
            attn_drop (float, optional): Dropout rate on attention weights. Defaults to 0.
            proj_drop (float, optional): Dropout rate on the final output. Defaults to 0.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # Scaling factor for dot products.
        self.head_dim = head_dim

        # Linear layers for query, key, and value projections.
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias) # Key and value are projected together for efficiency.
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Final projection layer.
        self.proj = nn.Linear(head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv, attn_mask=None):
        """The forward pass for the attention mechanism."""
        Bx, Nx, Cx = x.shape
        B, N, C = kv.shape

        # Project query and reshape for multi-head attention.
        q = self.q(x).reshape(Bx, Nx, self.num_heads, Cx//self.num_heads)
        q = q.permute(0, 2, 1, 3) # (B, H, N, C/H)

        # Project key and value together, then split and reshape.
        kv = self.kv(kv)
        kv = kv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4) # (2, B, H, N, C/H)
        k, v = kv.unbind(0) # Split into key and value.
        
        # Calculate scaled dot-product attention scores.
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N+1, C/H) @ (B, H, C/H, N+1) -> (B, H, N+1, N+1)

        # Apply softmax to get attention weights.
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute the weighted sum of values.
        x = attn @ v # (B, H, N_q, C/H)

        # Reshape and project back to the original dimension.
        x = x.transpose(1, 2).reshape(Bx, Nx, -1)  # (B, H, N+1, N+1) * (B, H, N+1, C/H) -> (B, H, N+1, C/H) -> (B, N+1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x