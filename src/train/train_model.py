import torch
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import torchvision 
from datetime import datetime

def train_model(
    model,
    model_output_path,
    dataloader_train,
    dataloader_eval,
    dataloader_test,
    num_epochs,
    optimizer,
    noise_scheduler,
    lr_scheduler,
    loss_fn,
    args
):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulate_grads_every_x_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.model_output_path, "logs"),
    )
    if accelerator.is_main_process:
        if args.model_output_path is not None:
            os.makedirs(args.model_output_path, exist_ok=True)
            models_dir = os.path.join(args.model_output_path, "models_pth")
            os.makedirs(models_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.model_output_path).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare the context for evaluation
    eval_noise = torch.randn(args.gen_eval_batch_size)
    eval_indices = torch.randint(0, len(dataloader_eval), (args.gen_eval_batch_size,))

    # Prepare the objects
    model, optimizer, dataloader_train, dataloader_eval, eval_noise, eval_indices, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader_train, dataloader_eval, eval_noise, eval_indices, lr_scheduler
    )

    def train_step(batch):
        model.train()

        clean_images = batch[0]

        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
            dtype=torch.int64
        )

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        with accelerator.accumulate(model):
            context = batch[1]
            noise_pred = model(noisy_images, timesteps, context, return_dict=False)[0]
            loss = loss_fn(noise_pred, noise)
            accelerator.backward(loss)

        return loss.item().detach()

    def eval_full():
        model.eval()
        eval_loss = []

        for step, batch in enumerate(tqdm(dataloader_eval)):
            clean_images = batch[0]

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            with torch.no_grad():
                context = batch[1]
                noise_pred = model(noisy_images, timesteps, context, return_dict=False)[0]
                loss = loss_fn(noise_pred, noise)
            eval_loss.append(loss.item().detach())
            
        return eval_loss, np.mean(eval_loss), np.sum(eval_loss)

    def generate_eval_step():
        model.eval()

        # Copy the x
        x = eval_noise.clone().detach() 

        # Now get the context and images
        clean_images, eval_context = dataloader_eval[eval_indices]

        # Sampling loop
        for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

            # Get model pred
            with torch.no_grad():
                residual = model(x, t, context=eval_context, return_dict=False)  # Again, note that we pass in our labels y

            # Update sample with step
            x = noise_scheduler.step(residual[0], t, x).prev_sample

        # Now return the x and images
        loss = loss_fn(x, clean_images)

        return x, clean_images, loss.item().detach()

    # Now let's iterate
    global_step = 1
    all_stats = {
        "train_loss": [],
        "train_epoch_loss": [],
        "eval_loss": [],
        "gen_eval_loss": []
    }
    display_stats = {
        "A - Batch Loss": 0,
        "B - Avg Loss": 0,
        "C - Epoch Loss": 0,
        "D - Val Loss": 0,
        "E - Gen Eval Loss": 0,
        "F - LR": 0,
    }

    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(dataloader_train), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = 0
        for step, batch in enumerate(dataloader_train):
            # Take a train step
            batch_loss = train_step(batch)

            # Update loss
            epoch_loss += batch_loss
            all_stats["train_loss"].append(batch_loss)
            display_stats["A - Batch Loss"] = batch_loss
            display_stats["B - Avg Loss"] = np.mean(all_stats["train_loss"][-1000:])

            # Now determine if we should eval
            if global_step % args.eval_metrics_every_x_batches:
                # Run eval
                eval_loss, mean_eval_loss, total_eval_loss = eval_full()

                # Update stats
                all_stats["eval_loss"].append(mean_eval_loss)
                display_stats["D - Val Loss"] = mean_eval_loss

            if global_step % args.gen_eval_every_x_batches:
                # Run gen eval
                gen_images, clean_images, gen_eval_loss = generate_eval_step()

                # Update stats
                all_stats["gen_eval_loss"].append(mean_eval_loss)
                display_stats["E - Gen Eval Loss"] = mean_eval_loss

            # Increment the global_step
            global_step += 1
            progress_bar.set_postfix(**display_stats)
            accelerator.log(display_stats, step=global_step)

        # Now update
        all_stats["train_epoch_loss"].append(epoch_loss / (step+1))
        display_stats["C - Epoch Loss"] = np.mean(all_stats["epoch_loss"][-1000:])

        if accelerator.is_main_process:
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                eval_loss, mean_eval_loss, total_eval_loss = eval_full()
                checkpoint_filename = f'{models_dir}/diffusion_checkpoint_epoch_{epoch}_{datetime.now().strftime("%Y-%m-%d")}_loss={mean_eval_loss}.pth'
                torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'eval_loss': mean_eval_loss,
                }, checkpoint_filename)