import torch
from torch import nn
from diffusers import UNet2DModel, UNet2DConditionModel

class StateEmbedding(nn.Module):
    def __init__(self, n_state, out_channels):
        super().__init__()
        self.linear_1 = nn.Linear(n_state, out_channels)
        self.linear_2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        # x: (1, 9)

        # (1, 9) -> (1, 9)
        x = self.linear_1(x)

        # (1, 9) -> (1, 9)
        x = nn.functional.silu(x)

        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        # (1, 1280) -> (1, 1280)
        x = nn.functional.silu(x)

        return x
        
class ClassConditionedUnet(nn.Module):
  def __init__(self, model, num_inputs, emb_size):
    super().__init__()
    
    # The embedding layer will map the class label to a vector of size class_emb_size
    self.state_emb = StateEmbedding(num_inputs, emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = model

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, context, return_dict=False):
    # Shape of x:
    bs, ch, w, h = x.shape
    
    # class conditioning in right shape to add as additional input channels
    context = self.state_emb(context) # Map to embedding dinemsion
    context = class_cond.view(bs, context.shape[1], 1, 1).expand(bs, context.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)
    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)

    # Feed this to the unet alongside the timestep and return the prediction
    return self.model(net_input, t, return_dict=return_dict)

# Load the model state
def load_model_state(model, path, optimizer=None):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  return epoch, loss

def load_class_conditional_model(image_size, context_size, emb_size):
  model = UNet2DModel(
      sample_size=image_size,           # the target image resolution
      in_channels=1 + emb_size, # Additional input channels for class cond.
      out_channels=1,           # the number of output channels
      layers_per_block=2,       # how many ResNet layers to use per UNet block
      block_out_channels=(32, 32, 64, 64),
      down_block_types=(
          "DownBlock2D",        # a regular ResNet downsampling block
          "DownBlock2D",
          "DownBlock2D",
          "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
          "AttnDownBlock2D",
      ),
      up_block_types=(
          "AttnUpBlock2D",
          "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
          "UpBlock2D",          # a regular ResNet upsampling block
          "UpBlock2D",
          "UpBlock2D",
        ),
      time_embedding_type='positional'
  )
  return ClassConditionedUnet(model, context_size, emb_size)

# Model Other - Not used
# class_embed_type="projection"
# num_class_embeds=4
# projection_class_embeddings_input_dim=128
# class_embeddings_concat=True
def load_cross_attention_model(image_size, context_size, emb_size):
  model = UNet2DConditionModel(
    sample_size=image_size,         # the target image resolution, as set above
    in_channels=1,            # Additional input channels for class cond.
    out_channels=1,           # the number of output channels
    layers_per_block=2,       # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 128),
    down_block_types=(
      "CrossAttnDownBlock2D",
      "CrossAttnDownBlock2D",
      "DownBlock2D"
    ),
    mid_block_type="UNetMidBlock2DCrossAttn",
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
      ),
    time_embedding_type='positional', # Or fourier
    dropout=0.1,
    cross_attention_dim=emb_size,
    encoder_hid_dim_type="text_proj",
    encoder_hid_dim=context_size
  )
  return model