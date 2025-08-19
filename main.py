import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from pathlib import Path
from models import FlexMoE
from utils import seed_everything, setup_logger
from data import create_loaders, collate_fn
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

def str2bool(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args():
    parser = argparse.ArgumentParser(description='FlexMoE for Multi-Modal Learning (regression, DataFrame-based)')
    parser.add_argument('--data', type=str, default='Kinetic_Params',
                        help='A name for the dataset, used for logging and loss function selection.')
    parser.add_argument('--data_dir', type=str, default='/home/adhil/MLRxnDB/data/datasets/kinetic_dataset/S4',
                        help='Directory containing train.parquet, val.parquet, and test.parquet files.')
    parser.add_argument('--label_col', type=str, default='log10_value',
                        help='Name of the label column in the Parquet files (for regression).')
    parser.add_argument('--modality', type=str, default='12345678',
                        help='String indicating which modalities to use (e.g., "12345678" for all 8).')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='The hidden dimension of the Transformer model.')
    parser.add_argument('--num_patches', type=int, default=16,
                        help='Number of patches to create from each input embedding.')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='Output dimension (1 for regression).')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer layers in the FlexMoE backbone.')
    parser.add_argument('--num_layers_pred', type=int, default=1,
                        help='Number of layers in the prediction head.')
    parser.add_argument('--num_experts', type=int, default=16,
                        help='Number of experts in the Mixture-of-Experts layers.')
    parser.add_argument('--num_routers', type=int, default=1,
                        help='Number of routers in the MoE layers.')
    parser.add_argument('--top_k', type=int, default=4,
                        help='Number of experts to route to for each token.')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for the model.')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID to use (default: 0).')
    parser.add_argument('--train_epochs', type=int, default=50,
                        help='Total number of training epochs.')
    parser.add_argument('--warm_up_epochs', type=int, default=5,
                        help='Number of warm-up epochs with curriculum learning.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training and evaluation.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--gate_loss_weight', type=float, default=1e-2,
                        help='Weight for the router load-balancing loss.')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of times to run the experiment with different seeds.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed.')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of workers for the DataLoader.')
    parser.add_argument('--pin_memory', type=str2bool, default=True,
                        help='Pin memory in DataLoader for faster GPU transfer (True/False).')
    parser.add_argument('--save', type=str2bool, default=True,
                        help='Whether to save the best model (True/False).')
    parser.add_argument('--load_model', type=str2bool, default=False,
                        help='Whether to load a pre-trained model for evaluation (True/False).')
    return parser.parse_known_args()

def run_epoch(args, loader, fusion_model, criterion, device, is_training=False, optimizer=None):
    all_preds = []
    all_labels = []
    losses = []
    modalities = [
        'smiles_1d_embedding', 'smiles_2d_embedding', 'smiles_3d_embedding',
        'smiles_clamp_embedding', 'ecfp', 'atom_pair_fp',
        'esmc_embedding', 'esm3_embedding'
    ]
    if is_training:
        fusion_model.train()
    else:
        fusion_model.eval()
    for batch_samples, batch_labels, batch_mcs, batch_observed in loader:
        batch_samples = {k: v.to(device) for k, v in batch_samples.items()}
        batch_labels = batch_labels.to(device)
        if is_training:
            optimizer.zero_grad()
            outputs = fusion_model(*[batch_samples[mod] for mod in modalities])
            loss = criterion(outputs.squeeze(), batch_labels.float())
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = fusion_model(*[batch_samples[mod] for mod in modalities])
                loss = criterion(outputs.squeeze(), batch_labels.float())
        losses.append(loss.item())
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(batch_labels.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return np.mean(losses), all_preds, all_labels

def train_and_evaluate(args, seed, save_path=None):
    seed_everything(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    modalities = [
        'smiles_1d_embedding', 'smiles_2d_embedding', 'smiles_3d_embedding',
        'smiles_clamp_embedding', 'ecfp', 'atom_pair_fp',
        'esmc_embedding', 'esm3_embedding'
    ]
    modality_dict = {mod: i for i, mod in enumerate(modalities)}

    train_df = pd.read_parquet(Path(args.data_dir) / "train.parquet")
    val_df = pd.read_parquet(Path(args.data_dir) / "val.parquet")
    test_df = pd.read_parquet(Path(args.data_dir) / "test.parquet")

    train_loader, val_loader, test_loader = create_loaders(
        train_df, val_df, test_df,
        modalities=modalities,
        label_col=args.label_col,
        modality_dict=modality_dict,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    train_loader_shuffle = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    input_dims = [train_df.dropna()[mod].iloc[0].shape[-1] for mod in modalities]

    fusion_model = FlexMoE(
        num_modalities=len(modalities),
        full_modality_index=0,
        num_patches=args.num_patches,
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_layers_pred=args.num_layers_pred,
        num_experts=args.num_experts,
        num_routers=args.num_routers,
        top_k=args.top_k,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()  # For regression

    best_val_loss = float('inf')
    best_model = None

    for epoch in trange(args.train_epochs):
        if epoch >= args.warm_up_epochs:
            train_loader_new = train_loader_shuffle
        else:
            train_loader_new = train_loader
        train_loss, _, _ = run_epoch(args, train_loader_new, fusion_model, criterion, device, is_training=True, optimizer=optimizer)
        val_loss, val_preds, val_labels = run_epoch(args, val_loader, fusion_model, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        fusion_model.clear_gate_buffers()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = fusion_model.state_dict()
            if args.save:
                os.makedirs('./saves', exist_ok=True)
                save_path = f'./saves/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}.pth'
                torch.save(best_model, save_path)
                print(f"Best model saved to {save_path}")

    # Test evaluation
    fusion_model.load_state_dict(best_model)
    test_loss, test_preds, test_labels = run_epoch(args, test_loader, fusion_model, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

def main():
    args, _ = parse_args()
    logger = setup_logger('./logs', f'{args.data}', f'{args.modality}.txt')
    seeds = np.arange(args.n_runs)
    for seed in seeds:
        train_and_evaluate(args, seed)

if __name__ == '__main__':
    main()