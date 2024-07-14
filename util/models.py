import torch
from torch import nn
from diffusers import UNet2DModel, UNet2DConditionModel

class StateEmbedding(nn.Module):
    def __init__(self, n_state, out_channels):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_state, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.linear(x)

class ClassConditionedUnet(nn.Module):
  def __init__(self, model, context_size, n_channels=0):
    super().__init__()

    # Get the time embedding
    self.time_embed_dim = n_channels*4
    
    # The embedding layer will map the class label to a vector of size class_emb_size
    self.state_emb = StateEmbedding(context_size, self.time_embed_dim)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = model

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels=None, return_dict=False):
    # Class conditioning
    context = self.state_emb(class_labels) # Map to embedding dinemsion

    # Feed this to the unet alongside the timestep and return the prediction
    return self.model(x, t, class_labels=context, return_dict=return_dict)

class InputConditionedUnet(nn.Module):
  def __init__(self, model):
    super().__init__()

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = model

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels=None, return_dict=False):
    # Shape of x:
    bs, ch, w, h = x.shape
    
    # Class conditioning in right shape to add as additional input channels
    context = class_labels.view(bs, -1, 1, 1).expand(bs, -1, w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)
    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, context), 1) # (bs, 6, 28, 28)

    # Feed this to the unet alongside the timestep and return the prediction
    return self.model(net_input, t, return_dict=return_dict)

def load_basic_diffusion_model(image_size):
  model = UNet2DModel(
      sample_size=image_size,  # the target image resolution
      in_channels=1,  # the number of input channels, 3 for RGB images
      out_channels=1,  # the number of output channels
      layers_per_block=2,  # how many ResNet layers to use per UNet block
      block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
      down_block_types=(
          "DownBlock2D",  # a regular ResNet downsampling block
          "DownBlock2D",
          "DownBlock2D",
          "DownBlock2D",
          "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
          "DownBlock2D",
      ),
      up_block_types=(
          "UpBlock2D",  # a regular ResNet upsampling block
          "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
      ),
  )
  return model

def load_class_diffusion_model(image_size, context_size):
  model = UNet2DModel(
      sample_size=image_size,  # the target image resolution
      in_channels=1,  # the number of input channels, 3 for RGB images
      out_channels=1,  # the number of output channels
      layers_per_block=2,  # how many ResNet layers to use per UNet block
      block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
      down_block_types=(
          "DownBlock2D",  # a regular ResNet downsampling block
          "DownBlock2D",
          "DownBlock2D",
          "DownBlock2D",
          "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
          "DownBlock2D",
      ),
      up_block_types=(
          "UpBlock2D",  # a regular ResNet upsampling block
          "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
      ),
      class_embed_type="identity"
  )
  return ClassConditionedUnet(model, context_size, n_channels=128)

def load_input_diffusion_model(image_size, context_size):
  model = UNet2DModel(
      sample_size=image_size+context_size,  # the target image resolution
      in_channels=1,  # the number of input channels, 3 for RGB images
      out_channels=1,  # the number of output channels
      layers_per_block=2,  # how many ResNet layers to use per UNet block
      block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
      down_block_types=(
          "DownBlock2D",  # a regular ResNet downsampling block
          "DownBlock2D",
          "DownBlock2D",
          "DownBlock2D",
          "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
          "DownBlock2D",
      ),
      up_block_types=(
          "UpBlock2D",  # a regular ResNet upsampling block
          "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
      ),
      class_embed_type="identity"
  )
  return InputConditionedUnet(model, context_size, n_channels=128)

# Load the model state
def load_model_state(model, path, optimizer=None):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  return epoch, loss

# def load_class_conditional_model(image_size, context_size, emb_size):
#   model = UNet2DModel(
#       sample_size=image_size,           # the target image resolution
#       in_channels=1 + emb_size, # Additional input channels for class cond.
#       out_channels=1,           # the number of output channels
#       layers_per_block=2,       # how many ResNet layers to use per UNet block
#       block_out_channels=(32, 32, 64, 64, 128),
#       down_block_types=(
#           "DownBlock2D",        # a regular ResNet downsampling block
#           "DownBlock2D",
#           "DownBlock2D",
#           "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
#           "AttnDownBlock2D",
#       ),
#       up_block_types=(
#           "AttnUpBlock2D",
#           "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
#           "UpBlock2D",          # a regular ResNet upsampling block
#           "UpBlock2D",
#           "UpBlock2D",
#         ),
#       time_embedding_type='positional'
#   )
#   return ClassConditionedUnet(model, context_size, emb_size)

# def load_class_conditional_model_class_emb(image_size, context_size, emb_size):
#     model = UNet2DModel(
#         sample_size=image_size,
#         in_channels=1, 
#         out_channels=1, 
#         layers_per_block=3,  # Increased depth per block
#         block_out_channels=(32, 64, 64, 128, 256),  # Increased width
#         down_block_types=(
#             "DownBlock2D",
#             "DownBlock2D",
#             "AttnDownBlock2D",
#             "AttnDownBlock2D",
#             "AttnDownBlock2D",
#         ),
#         up_block_types=(
#             "AttnUpBlock2D",
#             "AttnUpBlock2D",
#             "AttnUpBlock2D",
#             "UpBlock2D",
#             "UpBlock2D",
#         ),
#         time_embedding_type='positional',  # Changed to Fourier for better performance in some cases
#         class_embed_type='timestep',
#         num_class_embeds=context_size
#     )
#     return model

# Model Other - Not used
# class_embed_type="projection"
# num_class_embeds=4
# projection_class_embeddings_input_dim=128
# class_embeddings_concat=True
# def load_cross_attention_model(image_size, context_size, emb_size):
#   model = UNet2DConditionModel(
#     sample_size=image_size,         # the target image resolution, as set above
#     in_channels=1,            # Additional input channels for class cond.
#     out_channels=1,           # the number of output channels
#     layers_per_block=2,       # how many ResNet layers to use per UNet block
#     block_out_channels=(32, 64, 128),
#     down_block_types=(
#       "CrossAttnDownBlock2D",
#       "CrossAttnDownBlock2D",
#       "DownBlock2D"
#     ),
#     mid_block_type="UNetMidBlock2DCrossAttn",
#     up_block_types=(
#         "UpBlock2D",
#         "CrossAttnUpBlock2D",
#         "CrossAttnUpBlock2D",
#       ),
#     time_embedding_type='positional', # Or fourier
#     dropout=0.1,
#     cross_attention_dim=emb_size,
#     encoder_hid_dim_type="text_proj",
#     encoder_hid_dim=context_size
#   )
#   return model