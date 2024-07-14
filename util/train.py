from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch.nn.functional as F
import numpy as np

def generate(pipeline, seed):
  torch.manual_seed(seed)
  noise_scheduler = pipeline.scheduler
  net = pipeline.unet
  bs = 16
  x = torch.randn(bs, 1, size, size).to('cuda')

  # Sampling loop
  for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

      # Get model pred
      with torch.no_grad():
          residual = net(x, t, return_dict=False)  # Again, note that we pass in our labels y

      # Update sample with step
      x = noise_scheduler.step(residual[0], t, x).prev_sample

  # Show the results
  fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  ax.grid(False)
  ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=4)[0], cmap='Greys')
  return x

def evaluate(config, step, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    # images = pipeline(
    #     batch_size=config.eval_batch_size,
    #     generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    # ).images
    images = generate(pipeline, config.seed)

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4, resize=64)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/sample_{step:08d}.png")

def train(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    train_loss = []
    tot_epoch_loss = []
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
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
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            batch_loss = loss.detach().item()
            epoch_loss += batch_loss
            train_loss.append(batch_loss)
            logs = {"Batch Loss": batch_loss, "Epoch Loss": epoch_loss / (step+1), "Avg Loss": np.mean(train_loss[-1000:]), "LR": lr_scheduler.get_last_lr()[0], "Step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # Check if we should output
            if (global_step + 1) % 2000 == 0:
              pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
              evaluate(config, global_step, pipeline)

        tot_epoch_loss.append(epoch_loss / step)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, global_step, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
              models_dir = os.path.join(config.output_dir, "models_pth")
              os.makedirs(models_dir, exist_ok=True)
              current_date = datetime.now().strftime("%Y-%m-%d")
              checkpoint_filename = f'{models_dir}/diffusion_checkpoint_epoch_{epoch}_{current_date}.pth'
              torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': epoch_loss,
              }, checkpoint_filename)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)
    return model, train_loss, tot_epoch_loss