import wandb
import argparse
from models import FlexMoE
import pytorch_lightning as pl
from datamodule import FlexMoEDataModule
from pytorch_lightning.loggers import WandbLogger

def str2bool(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args():
    parser = argparse.ArgumentParser(description='FlexMoE for Multi-Modal Learning (regression, DataFrame-based)')
    parser.add_argument('--data_dir', type=str, default='/home/adhil/MLRxnDB/data/datasets/kinetic_dataset/S4')
    parser.add_argument('--label_col', type=str, default='log10_value')
    parser.add_argument('--modality', type=str, default='12345678')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_patches', type=int, default=16)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_layers_pred', type=int, default=1)
    parser.add_argument('--num_experts', type=int, default=16)
    parser.add_argument('--num_routers', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gate_loss_weight', type=float, default=1e-2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str2bool, default=False)
    return parser.parse_args()

def train_model(config=None):
    # For wandb sweeps: config is provided by wandb
    with wandb.init(config=config):
        config = wandb.config if config is None else config

        pl.seed_everything(config.seed)

        modalities = [
            'smiles_1d_embedding', 'smiles_2d_embedding', 'smiles_3d_embedding',
            'smiles_clamp_embedding', 'ecfp', 'atom_pair_fp',
            'esmc_embedding', 'esm3_embedding'
        ]
        modality_dict = {mod: i for i, mod in enumerate(modalities)}

        datamodule = FlexMoEDataModule(
            data_dir=config.data_dir,
            modalities=modalities,
            label_col=config.label_col,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            modality_dict=modality_dict
        )
        datamodule.setup()
        input_dims = datamodule.get_input_dims()

        model = FlexMoE(
            num_modalities=len(modalities),
            full_modality_index=0,
            num_patches=config.num_patches,
            input_dims=input_dims,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            num_layers_pred=config.num_layers_pred,
            num_experts=config.num_experts,
            num_routers=config.num_routers,
            top_k=config.top_k,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

        wandb_logger = WandbLogger(
            project=config.wandb_project,
            log_model=True
        )

        trainer = pl.Trainer(
            max_epochs=config.train_epochs,
            accelerator="auto",
            devices="auto",
            logger=wandb_logger,
            log_every_n_steps=10,
            deterministic=True,
            precision="16-mixed",
            default_root_dir="./saves" if config.save else None,
        )

        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

def main():
    args = parse_args()

    sweep_config = {
        "method": "bayes",  # or "random", "grid"
        "metric": {
            "name": "val/mse",  # This should match your logged metric
            "goal": "minimize"
        },
        "parameters": {
            "hidden_dim": {"values": [128, 256, 512]},
            "num_layers": {"values": [2, 4, 6]},
            "lr": {"values": [1e-3, 5e-4, 1e-4]},
            "batch_size": {"values": [1024]},
            "dropout": {"values": [0.3, 0.5]},
            "num_heads": {"values": [2, 4, 6]},
            "num_experts": {"values": [8, 16]},
            "top_k": {"values": [2, 4]},
            "train_epochs": {"value": args.train_epochs},
            "data_dir": {"value": args.data_dir},
            "label_col": {"value": args.label_col},
            "num_patches": {"value": args.num_patches},
            "output_dim": {"value": args.output_dim},
            "num_layers_pred": {"value": args.num_layers_pred},
            "num_routers": {"value": args.num_routers},
            "num_workers": {"value": args.num_workers},
            "pin_memory": {"value": args.pin_memory},
            "seed": {"value": args.seed},
            "save": {"value": args.save},
            "wandb_project": {"value": "FlexMoE"},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="FlexMoE")
    print(f"Starting sweep with ID: {sweep_id}")
    wandb.agent(sweep_id, function=train_model, count=50)  # Set count as needed

if __name__ == '__main__':
    main()