import torch
import torchvision
from torch import nn
from diffusers import DDPMScheduler
from datetime import datetime
import os
from auto import tqdm
from matplotlib import pyplot as plt

def get_device():
  return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def train_loop(
  model,
  train_dataloader,
  model_output_path,
  n_epochs,
  noise_scheduler=None,
  loss_fn=None,
  opt=None, 
  valid_dataloader=None,
  gen_data=None
):
  # Get the device
  device = get_device()

  # Get the net
  net = model.to(device)

  # Set the noise scheduler if none
  if noise_scheduler == None:
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

  # Set the loss function if none
  if loss_fn == None:
    loss_fn = nn.MSELoss()

  # Set the optimizer if none
  if opt == None:
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

  # Define location for model output
  current_date = datetime.now().strftime("%Y-%m-%d")
  if not os.path.exists(model_output_path):
      os.makedirs(model_output_path)

  # Track loss values
  train_loss = []
  valid_loss   = []

  # The training loop
  for epoch in range(n_epochs):
    net.train()
    epoch_loss = 0
    step = 0
    for x, y in tqdm(train_dataloader):
        # Get some data and prepare the corrupted version
        x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        pred = net(noisy_x, timesteps, y, return_dict=False) # Note that we pass in the labels y

        # Calculate the loss
        loss = loss_fn(pred[0], noise) # How close is the output to the noise

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Increment the loss
        epoch_loss += loss.item()

        # Step
        step += 1
    
    # Now update loss and check to store model output
    train_loss.append(epoch_loss / (step + 1))

    # Print out the epoch loss
    print(f'{datetime.now()} - Finished Training Epoch: {epoch}. Training Epoch Loss: {train_loss[-1]}')

    # Store the model
    if epoch % 1 == 0:
      print(f"{datetime.now()} - Saving model")
      checkpoint_filename = f'{model_output_path}/diffusion_checkpoint_epoch_{epoch}_{current_date}.pth'
      torch.save({
          'epoch': epoch,
          'model_state_dict': net.state_dict(),
          'optimizer_state_dict': opt.state_dict(),
          'loss': epoch_loss,
      }, checkpoint_filename)

    # Run validation loop
    if epoch % 1 == 0 and valid_dataloader is not None:
      print(f"{datetime.now()} - Running validation")
      val_epoch_loss, _, _ = valid_loop(
        net,
        valid_dataloader,
        noise_scheduler,
        loss_fn
      )
      # Now update loss and check to store model output
      valid_loss.append(val_epoch_loss)

      # Print out the epoch loss
      print(f'{datetime.now()} - Finished Validation Run: {epoch}. Validation Epoch Loss: {valid_loss[-1]}')
    
    if epoch % 1 == 0 and gen_data is not None:
      print(f"{datetime.now()} - Running generation")
      figure_path = f'{model_output_path}/gen_figure_{epoch}_{current_date}.png'
      gen_loop(
        net,
        gen_data,
        figure_path,
        noise_scheduler
      )

  return net, train_loss, valid_loss 

def valid_loop(net, valid_dataloader, noise_scheduler=None, loss_fn=None):
  # Get the device
  device = get_device()

  # Set the noise scheduler if none
  if noise_scheduler == None:
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
  
  # Set the loss function if none
  if loss_fn == None:
    loss_fn = nn.MSELoss()

  # Now run a validation step
  net.eval()
  val_epoch_loss = 0
  val_step = 0
  for x, y in tqdm(valid_dataloader):
    with torch.no_grad():
      # Get some data and prepare the corrupted version
      x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
      y = y.to(device)
      noise = torch.randn_like(x)
      timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
      noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

      # Get the model prediction
      pred = net(noisy_x, timesteps, y, return_dict=False) # Note that we pass in the labels y

      # Calculate the loss
      val_loss = loss_fn(pred[0], noise) # How close is the output to the noise
      val_epoch_loss += val_loss.item()

      # Update the step
      val_step += 1
  return val_epoch_loss, val_epoch_loss / (val_step + 1), val_step

def gen_loop(net, gen_data, figure_path, noise_scheduler=None):
  # Get the device
  device = get_device()

  # Get the data
  x, y = gen_data
  x = x.to(device)
  y = y.to(device)

  # Set the noise scheduler if none
  if noise_scheduler == None:
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

  # Sampling loop
  net.eval()
  for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

      # Get model pred
      with torch.no_grad():
          residual = net(x, t, y, return_dict=True).sample  # Again, note that we pass in our labels y

      # Update sample with step
      x = noise_scheduler.step(residual, t, x).prev_sample

  # Show the results
  fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=4)[0], cmap='Greys')
  
  # Save and close
  fig.savefig(figure_path)
  plt.close(fig)
