from diffusers import UNet2DModel, UNet2DConditionModel

def load_class_conditional_model(image_size, context_size, emb_size):
  model = UNet2DModel(
      sample_size=image_size,           # the target image resolution
      in_channels=1 + emb_size, # Additional input channels for class cond.
      out_channels=1,           # the number of output channels
      layers_per_block=2,       # how many ResNet layers to use per UNet block
      block_out_channels=(64, 128, 128, 256, 512),
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