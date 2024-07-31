import argparse
from argparse import Namespace
import torch
import pandas as pd
import numpy as np
import json
import os
import re
from torch.utils.data import DataLoader
import sys
import yaml

# sys.path.insert(0, "../model/")
# From chexchonet
from datasets import (collate_cxr, CXRInferenceDataset)
from train_model import train_model
from models import (
    DenseNet121Base,
    DenseNet121MultiView,
    DenseNet121BaseWithDemo,
    EfficientNetDynamic,
)
from libauc.optimizers import PESG, Adam

# From cxr_diffusion
# from cxr_diffusion.src.train.datasets import CXRInferenceDataset

def run(args):
  if(args.cuda_idx >= 0) and torch.cuda.is_available():
      device = torch.device("cuda:{}".format(args.cuda_idx))
      dataloader_kwargs = {"pin_memory": True, "num_workers": 8}
      print(device)
  else:
      device = torch.device("cpu")
      dataloader_kwargs = {}

  args.device = device

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
  dset_class = CXRInferenceDataset
  dataset_train = dset_class(**args.train_dataset_settings)
  args.eval_dataset_settings["age_stats"] = (
      dataset_train.age_mean,
      dataset_train.age_var,
  )
  args.eval_dataset_settings["enc"] = dataset_train.enc
  args.test_dataset_settings["age_stats"] = (
      dataset_train.age_mean,
      dataset_train.age_var,
  )
  args.test_dataset_settings["enc"] = dataset_train.enc

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

  label_weights = None
  if not args.continuous_labels:
      temp_df = pd.read_csv(args.train_dataset_settings["metadata_df_path"])
      label_weights = []

      for label_col in args.train_dataset_settings["label_cols"]:
          class_counts = temp_df[label_col].value_counts().tolist()
          weights = torch.tensor(np.array(class_counts) / sum(class_counts))
          print("CLASS 0: {}, CLASS 1: {}".format(weights[0], weights[1]))
          weights = weights[0] / weights
          print("WEIGHT 0: {}, WEIGHT 1: {}".format(weights[0], weights[1]))
          label_weights.append(weights[1])
      label_weights = torch.stack(label_weights)

  if args.optimizer == "adam":
      optimizer = torch.optim.Adam
  elif args.optimizer == "adamw":
      optimizer = torch.optim.AdamW
  elif args.optimizer == "sgd":
      optimizer = torch.optim.SGD

  model = None
  if args.model_name.startswith("dense_base"):

      if args.model_name.endswith("with_demo"):
          if args.continuous_labels and args.include_uncertainty:
              model = DenseNet121BaseWithDemo(
                  args.num_labels * 2,
                  demo_size=args.demo_size,
                  model_kwargs=args.model_kwargs,
              )
          else:
              model = DenseNet121BaseWithDemo(
                  args.num_labels,
                  demo_size=args.demo_size,
                  model_kwargs=args.model_kwargs,
              )
      else:
          if args.continuous_labels and args.include_uncertainty:
              model = DenseNet121Base(
                  args.num_labels * 2, model_kwargs=args.model_kwargs
              )
          else:
              model = DenseNet121Base(args.num_labels, model_kwargs=args.model_kwargs)

      if args.use_chexnet_pretrained:
          # chexnet pretrained weights from https://github.com/arnoweng/CheXNet
          print("Loading CheXNet weights...")
          model_dict = model.state_dict()
          checkpoint = torch.load(args.ckpt_path, map_location=device)["state_dict"]

          pattern = re.compile(
              r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
          )
          for key in list(checkpoint.keys()):
              res = pattern.match(key)
              if res:
                  new_key = res.group(1) + res.group(2)
                  checkpoint[new_key] = checkpoint[key]
                  del checkpoint[key]

          pretrained_dict = {
              k.replace("module.densenet121", "densenet121"): v
              for k, v in checkpoint.items()
              if k.startswith("module.densenet121.features")
          }
          model_dict.update(pretrained_dict)
          model.load_state_dict(model_dict)
          print("Loaded successfully")
      if args.use_lvhnet_pretrained:
          print("Loading lvhnet weights...")
          model_dict = model.state_dict()
          checkpoint = torch.load(args.ckpt_path)["model"]

          model.load_state_dict(checkpoint)

  elif args.model_name == "efficient_base":
      model = EfficientNetDynamic(
          args.num_labels, args.input_size, pretrained=args.use_chexnet_pretrained
      )

  metrics = train_model(
      model,
      args.model_name,
      dataloader_train,
      dataloader_eval,
      dataloader_test,
      args.num_epochs,
      optimizer,
      args.optimizer_kwargs,
      args.class_to_name,
      args.class_to_name_continuous,
      args.input_size,
      args.num_labels,
      args.include_demo,
      args.continuous_labels,
      args.include_uncertainty,
      train_metrics_every_x_batches=args.train_metrics_every_x_batches,
      eval_metrics_every_x_batches=args.eval_metrics_every_x_batches,
      label_to_weight=label_weights,
      visdom_run_name=args.visdom_run_name,
      accumulate_grads_every_x_steps=args.accumulate_grads_every_x_steps,
      use_auroc_loss=args.use_auroc_loss,
      device=device,
      args=args,
  )

  return metrics

def main(file_path, train_file_path, test_file_path, name):
    with open(file_path, "r") as config_file:
        config_dict = yaml.safe_load(config_file)

    # Defaults
    config_dict['cuda_idx'] = 0
    config_dict['model_name'] = 'dense_base_with_demo'

    # Adjust 
    config_dict['metadata_df_paths']['train_metadata_path'] = train_file_path
    config_dict['metadata_df_paths']['test_metadata_path']  = test_file_path
    config_dict['visdom_run_name'] = name
    import json
    print(json.dumps(config_dict, indent=2))

    # Now convert
    args = Namespace(**config_dict)

    run(args)