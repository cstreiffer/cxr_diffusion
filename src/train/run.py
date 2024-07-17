import torch
from torch.utils.data import DataLoader
from torch import nn
from datasets import (
    CXRDiffusionDataset,
    collate_cxr
)
from models import (
    load_input_diffusion_model,
    load_class_diffusion_model,
    load_basic_diffusion_model
)
from loss import (
    perceptual_loss
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
from train_model import train_model
import os
from datetime import datetime

def run(args):
    if(args.cuda_idx >= 0) and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.cuda_idx))
        dataloader_kwargs = {"pin_memory": True, "num_workers": 8}
        print(device)
    else:
        device = torch.device("cpu")
        dataloader_kwargs = {}
    args.device = device

    # Define the output path
    args.model_output_path = os.path.join(
        args.model_output_path, 
        f'{args.model_name}_{args.dataset_settings["downsample_size"]}_{datetime.now().strftime("%Y-%m-%d")}'
    )

    # Load the datasets
    args.train_dataset_settings = args.dataset_settings.copy()
    args.train_dataset_settings["metadata_df_path"] = args.metadata_df_paths[
        "train_metadata_path"
    ]
    args.eval_dataset_settings = args.dataset_settings.copy()
    args.eval_dataset_settings["metadata_df_path"] = args.metadata_df_paths[
        "eval_metadata_path"
    ]
    args.eval_dataset_settings["train"] = False
    args.test_dataset_settings = args.dataset_settings.copy()
    args.test_dataset_settings["metadata_df_path"] = args.metadata_df_paths[
        "test_metadata_path"
    ]
    args.test_dataset_settings["train"] = False

    # Load the data frames
    dset_class = CXRDiffusionDataset
    dataset_train = dset_class(**args.train_dataset_settings)
    args.eval_dataset_settings["age_stats"] = (
        dataset_train.age_mean,
        dataset_train.age_var,
    )
    args.test_dataset_settings["age_stats"] = (
        dataset_train.age_mean,
        dataset_train.age_var,
    )
    dataset_eval = dset_class(**args.eval_dataset_settings)
    dataset_test = dset_class(**args.test_dataset_settings)

    dataloader_train = DataLoader(
        dataset_train,
        args.batch_size,
        shuffle=True,
        collate_fn=collate_cxr,
        **dataloader_kwargs
    )

    dataloader_eval = DataLoader(
        dataset_eval,
        args.batch_size,
        shuffle=False,
        collate_fn=collate_cxr,
        **dataloader_kwargs
    )

    dataloader_test = DataLoader(
        dataset_test,
        args.batch_size,
        shuffle=False,
        collate_fn=collate_cxr,
        **dataloader_kwargs
    )

    # Load the model
    model = None
    if args.model_name == "basic_diffusion":
        model = load_basic_diffusion_model(args.dataset_settings['downsample_size'], args.num_feats)
    elif args.model_name == "class_diffusion":
        model = load_class_diffusion_model(args.dataset_settings['downsample_size'], args.num_feats)
    elif args.model_name == "input_diffusion":
        model = load_input_diffusion_model(args.dataset_settings['downsample_size'], args.num_feats)

    # Place the model
    model = model.to(device)

    # Define the optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD

    # Now load the optimizer
    optimizer = optimizer(params=model.parameters(), **args.optimizer_kwargs)

    # Define the noise scheduler
    noise_scheduler = DDPMScheduler(**args.noise_scheduler_kwargs)

    # Define the learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader_train) * args.num_epochs),
    )

    # Define the loss function
    if args.loss_fn == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss_fn == 'mse_p':
        loss_fn = perceptual_loss()