import tree

import torch
import torch.nn as nn
import torch.nn.functional as F

from fmoe.transformer import _Expert
from fmoe.gates import NoisyGate, NaiveGate
from fmoe.functions import ensure_comm, Slice, AllGather
from fmoe.layers import _fmoe_general_global_forward, mark_module_parallel_comm

class FixedFMoE(nn.Module):
    """
    A foundational Mixture-of-Experts (MoE) layer with a fixed number of experts.

    This class implements the core logic for a general MoE layer. It handles the
    routing of input tokens to different "expert" sub-networks and combines their
    outputs. It is designed to be flexible and supports various configurations for
    distributed training, including model parallelism (slicing) and data parallelism.
    It also includes functionality for handling missing data through masking.
    """
    def __init__(self, num_expert=32, d_model=1024, world_size=1, mp_group=None, slice_group=None, moe_group=None, top_k=2, gate=NaiveGate, expert=None, gate_hook=None, mask=None, mask_dict=None):
        """
        Initializes the FixedFMoE layer.

        Args:
            num_expert (int): The total number of expert networks.
            d_model (int): The dimensionality of the input and output tensors.
            world_size (int): The total number of GPUs in the distributed training setup.
            mp_group: (Deprecated) The model parallel group. Use slice_group instead.
            slice_group: The model parallel group for slicing the model across GPUs.
            moe_group: The data parallel group for the MoE layer.
            top_k (int): The number of experts to route each token to.
            gate (nn.Module): The gating network class to use for routing decisions.
            expert (nn.Module, optional): The expert network class. If None, a fused expert is assumed.
            gate_hook (callable, optional): A hook function to be called with gating information.
            mask (torch.Tensor, optional): A tensor indicating which tokens to mask (e.g., for missing modalities).
            mask_dict (dict, optional): A dictionary of values to use for masked tokens.
        """
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()
        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            # If no expert class is provided, assume a single, fused expert module.
            self.experts_fused = True
        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def expert_fn(self, inp, fwd_expert_count):
        """
        Applies the expert networks to the input tensor.

        This function routes slices of the input tensor to their corresponding
        expert networks based on the `fwd_expert_count`.

        Args:
            inp (torch.Tensor): The input tensor after being dispatched to experts.
            fwd_expert_count (torch.Tensor or np.ndarray): A tensor containing the
                number of tokens routed to each expert.

        Returns:
            torch.Tensor: The concatenated output of all expert networks.
        """
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def mark_parallel_comm(self, expert_dp_comm="none"):
        """
        Marks modules for specific communication patterns in distributed training.

        This is a helper function for the distributed framework to identify which
        parts of the model (gate, experts) require which type of communication.

        Args:
            expert_dp_comm (str): The communication pattern for expert data parallelism.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp, expert_indices=None):
        """
        The forward pass for the MoE layer.

        This method orchestrates the entire MoE process:
        1. Handles distributed communication and slicing.
        2. Gets routing decisions (top-k indices and scores) from the gate.
        3. Optionally masks out certain tokens (e.g., for missing modalities).
        4. Dispatches tokens to the appropriate experts using `_fmoe_general_global_forward`.
        5. Reconstructs the tensor if masking was applied.
        6. Combines the expert outputs weighted by the gate scores.
        7. Gathers results across all GPUs if model parallelism is used.

        Args:
            moe_inp (torch.Tensor): The input tensor to the MoE layer.
            expert_indices (torch.Tensor, optional): A tensor providing specific
                expert indices for supervised routing, used by `AddtionalNoisyGate`.

        Returns:
            torch.Tensor: The output tensor after being processed by the MoE layer.
        """
        # Ensure all input tensors in the tree have the same batch size.
        moe_inp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_inp))
        assert all([batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]), "MoE inputs must have the same batch size"

        # Ensure communication buffers are ready for distributed training.
        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)
            tree.map_structure(ensure_comm_func, moe_inp)

        # Slice the input for model parallelism if world_size > 1.
        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)
            moe_inp = tree.map_structure(slice_func, moe_inp)

        # Get routing decisions from the gating network.
        gate_top_k_idx, gate_score = self.gate(moe_inp, expert_indices)

        # Reshape gate_top_k_idx to be 2-dimensional (batch_size, top_k).
        gate_top_k_idx = gate_top_k_idx.view(moe_inp.shape[0], self.top_k)

        # Optional hook for inspecting gate decisions.
        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # Store the chosen expert indices in the gate module for later access (e.g., for loss calculation).
        self.gate.set_topk_indicates(gate_top_k_idx)

        # If a mask is provided, remove the masked tokens before the expert computation.
        if self.mask is not None and self.mask_dict is not None:
            def delete_mask_func(tensor):
                tensor = tensor[self.mask == 0, :]
                return tensor
            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        # Core MoE computation: dispatch, expert forward pass, and combine.
        fwd = _fmoe_general_global_forward(moe_inp, gate_top_k_idx, self.expert_fn, self.num_expert, self.world_size, experts=self.experts)

        # If masking was used, recover the original tensor shape, filling in masked positions.
        if self.mask is not None and self.mask_dict is not None:
            def recover_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                x = torch.zeros(mask.shape[0], self.top_k, dim, device=tensor.device, dtype=tensor.dtype)
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x
            moe_outp = tree.map_structure(recover_func, fwd)
        else:
            # If no masking, just reshape the output.
            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor
            moe_outp = tree.map_structure(view_func, fwd)

        # Weight the expert outputs by the gate scores.
        gate_score = gate_score.view(-1, 1, self.top_k)
        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor
        moe_outp = tree.map_structure(bmm_func, moe_outp)

        # If using model parallelism, gather the results from all slices.
        if self.slice_size > 1:
            def all_gather_func(tensor):
                return AllGather.apply(tensor, self.slice_rank, self.slice_size, self.slice_group)
            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        # Final check to ensure output batch size is consistent.
        moe_outp_batch_size = tree.flatten(tree.map_structure(lambda tensor: tensor.shape[0], moe_outp))
        assert all([batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]), "MoE outputs must have the same batch size"
        return moe_outp


class FMoETransformerMLP(FixedFMoE):
    """
    A complete MoE MLP module designed to be a drop-in replacement for the
    feed-forward network (FFN) in a Transformer block.

    This class wraps the `FixedFMoE` layer, providing a standard MLP structure
    (two linear layers with an activation in between) within each expert. It also
    handles the necessary reshaping of the input tensor from a 3D Transformer
    format (seq_len, batch_size, d_model) to a 2D format for the MoE layer,
    and then back again.
    """
    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        n_router = 1,
        gate='AddtionalNoisyGate', # NaiveGate
        world_size=1,
        top_k=2,
        **kwargs
    ):
        """
        Initializes the FMoETransformerMLP layer.

        Args:
            num_expert (int): The total number of expert networks.
            d_model (int): The dimensionality of the input and output tensors.
            d_hidden (int): The hidden dimension of the MLP inside each expert.
            activation (nn.Module): The activation function to use in the MLP.
            expert_dp_comm (str): The communication pattern for expert data parallelism.
            expert_rank (int): The rank of the expert in distributed settings.
            n_router (int): The number of gating networks (routers) to create.
            gate (str or nn.Module): The gating network class to use.
            world_size (int): The total number of GPUs in the distributed training setup.
            top_k (int): The number of experts to route each token to.
            **kwargs: Additional arguments passed to the FixedFMoE parent class.
        """
        if type(gate) is str:
            gate = eval(gate)
        super().__init__(num_expert=num_expert, d_model=d_model, gate=gate, world_size=world_size, top_k=top_k, **kwargs)
        # Define the expert network structure, which is a standard two-layer MLP.
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.n_router = n_router
        # Create a dictionary of gating networks, allowing for multiple routers.
        self.all_gates = nn.ModuleDict({f'{i}': gate(d_model, num_expert, world_size, top_k) for i in range(n_router)})
        # Set the default gate to the first one.
        self.gate = self.all_gates[f'{0}']

        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor, expert_indices=None):
        """
        The forward pass for the Transformer MLP block.

        Args:
            inp (torch.Tensor): The input tensor from the Transformer, typically of
                shape (sequence_length, batch_size, d_model).
            expert_indices (torch.Tensor, optional): A tensor providing specific
                expert indices for supervised routing.

        Returns:
            torch.Tensor: The output tensor with the same shape as the input.
        """
        # Store original shape to reshape back at the end.
        original_shape = inp.shape
        # Reshape input from 3D to 2D for the MoE layer.
        inp = inp.reshape(-1, self.d_model)
        
        # Call the parent class's forward method to perform the MoE computation.
        output = super().forward(inp, expert_indices=expert_indices)

        # Reshape the output back to the original 3D shape.
        return output.reshape(original_shape)

    def set_full_modality(self, is_full_modality):
        """
        Informs the gating networks whether the current batch consists of samples
        that have all modalities present.

        This is a key part of the Flex-MoE dual-router system, allowing the gate
        to adapt its loss calculation for the G-Router (full modality) vs. the
        S-Router (missing modalities).

        Args:
            is_full_modality (bool): True if the batch contains only complete samples.
        """
        for gate in self.all_gates.values():
            if hasattr(gate, 'set_full_modality'):
                gate.set_full_modality(is_full_modality)


class AddtionalNoisyGate(NoisyGate):
    """
    A custom noisy gate designed for the Flex-MoE architecture.

    This gate extends the standard `NoisyGate` to implement the dual-router logic
    of Flex-MoE. It calculates two types of losses:
    1. A cross-entropy loss for the Specialized Router (S-Router), which learns to
       route samples with missing modalities to specific, designated experts.
    2. A load-balancing loss, which is calculated separately for the Generalized
       Router (G-Router) and the S-Router to encourage expert utilization.
    """
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        """
        Initializes the AddtionalNoisyGate.

        Args:
            d_model (int): The dimensionality of the input tensors.
            num_expert (int): The total number of expert networks.
            world_size (int): The total number of GPUs in the distributed training setup.
            top_k (int): The number of experts to route each token to.
        """
        super().__init__(d_model, num_expert, world_size, top_k)
        self.topk_logits = []
        self.indicates = None
        self.is_full_modality = False

    def set_topk_logit(self, logit):
        """Stores the logits of the top-k experts."""
        if self.topk_logits is None:
            self.topk_logits = []
        self.topk_logits.append(logit)
    
    def get_topk_logit(self, clear = True):
        """Retrieves and optionally clears the stored top-k logits."""
        topk_logit = self.topk_logits
        if clear:
            self.topk_logits = None
        return topk_logit

    def set_topk_indicates(self, indicate):
        """Stores the indices of the top-k experts chosen by the gate."""
        self.indicates = indicate
        
    def get_topk_indicate(self, clear = True):
        """Retrieves and optionally clears the stored top-k indices."""
        topk_indicate = self.indicates
        if clear:
            self.indicates = None
        return topk_indicate
    
    def set_loss(self, loss):
        """Accumulates the loss calculated by the gate during the forward pass."""
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss

    def clear_loss(self):
        self.loss = None
    
    def set_full_modality(self, is_full_modality):
        """Sets a flag indicating if the current batch is for the G-Router."""
        self.is_full_modality = is_full_modality

    def forward(self, inp, expert_indices=None):
        """
        The forward pass for the custom gate.

        This method calculates routing decisions and the associated losses for the
        Flex-MoE framework.

        Args:
            inp (torch.Tensor): The input tensor.
            expert_indices (torch.Tensor, optional): A tensor of target expert indices
                for the S-Router. Its presence triggers the specialized loss calculations.

        Returns:
            tuple: A tuple containing:
                - The chosen top-k expert indices for each token.
                - The corresponding gate scores (probabilities) for each chosen expert.
        """
        # Standard noisy gate logic: calculate logits with added noise.
        clean_logits = inp @ self.w_gate
        raw_noise_stddev = inp @ self.w_noise
        noise_stddev = (
            self.softplus(raw_noise_stddev) + self.noise_epsilon
        ) * self.training
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits
        loss = 0

        # Determine the top-k experts based on the noisy logits.
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        # This block contains the core logic for the S-Router (Specialized Router).
        # It is triggered only when `expert_indices` are provided for samples with missing modalities.
        if expert_indices is not None and expert_indices.sum() > 0:
            batch_size = inp.shape[0]
            num_experts = expert_indices.shape[0]
            
            # Expand the expert_indices to match the batch size of the input tensor.
            repeats = batch_size // num_experts
            remainder = batch_size % num_experts

            if repeats > 0:
                expert_indices_expanded = expert_indices.repeat(repeats, 1).T.reshape(-1)
            else:
                expert_indices_expanded = torch.tensor([], dtype=expert_indices.dtype, device=expert_indices.device)

            if remainder > 0:
                expert_indices_expanded = torch.cat([expert_indices_expanded, torch.tensor([expert_indices[-1]]*remainder).to(expert_indices.device)])
            
            # A mask to distinguish between G-Router samples (index 0) and S-Router samples.
            full_modality_mask_expanded = expert_indices_expanded == 0

            # Calculate the cross-entropy loss for the S-Router. This trains the gate
            # to route incomplete samples to their designated experts.
            expert_idx_loss = F.cross_entropy(logits[~full_modality_mask_expanded], expert_indices_expanded[~full_modality_mask_expanded])
            loss += expert_idx_loss
        
        self.set_topk_logit(top_k_indices)

        # Scatter the top-k gate scores into a full-size tensor for loss calculation.
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # Calculate the 'load' for each expert, which is part of the load-balancing loss.
        if self.top_k < self.tot_expert and self.training:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            )
        else:
            load = self._gates_to_load(gates)
        
        # This block calculates the load-balancing loss (cv_squared) separately for
        # the G-Router and S-Router based on the `expert_indices`.
        if (expert_indices != None):
            full_modality_mask = expert_indices == 0
            # Case 1: All samples in the batch have full modalities (G-Router only).
            if full_modality_mask.sum() == len(full_modality_mask):
                load = load.sum(0) if self.training else load
                importance = gates.sum(0) if self.training else gates.sum(0)
                loss += self.cv_squared(importance) + self.cv_squared(load)
            # Case 2: The batch contains a mix of full and incomplete samples.
            else:
                # Calculate loss for the G-Router samples.
                importance_1 = gates[full_modality_mask_expanded, :].sum(0) if self.training else gates.sum(0)
                load_1 = load[full_modality_mask_expanded, :].sum(0) if self.training else load
                loss_1 = self.cv_squared(importance_1) + self.cv_squared(load_1)

                # Calculate loss for the S-Router samples (ignoring the first expert, which is for G-Router).
                importance_2 = gates[~full_modality_mask_expanded, 1:].sum(0) if self.training else gates.sum(0)
                load_2 = load[~full_modality_mask_expanded, 1:].sum(0) if self.training else load
                loss_2 = self.cv_squared(importance_2) + self.cv_squared(load_2)

                loss = loss + loss_1 + loss_2
        else:
            # Default case if no expert_indices are provided (standard MoE behavior).
            load = load.sum(0) if self.training else load
            importance = gates.sum(0) if self.training else gates.sum(0)
            loss += self.cv_squared(importance) + self.cv_squared(load)
        
        # Store the final calculated loss in the module.
        self.set_loss(loss)
        
        # Return the chosen expert indices and their scores.
        return (
            top_k_indices.contiguous().view(-1),
            top_k_gates.contiguous().unsqueeze(1),
        )