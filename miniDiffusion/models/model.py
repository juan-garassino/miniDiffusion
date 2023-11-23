"""
Advanced U-Net Architecture for Deep Learning Applications

This Python file presents an implementation of an advanced U-Net model, a widely-used
architecture in deep learning, particularly for tasks like image segmentation and
generative models. The U-Net architecture is known for its effectiveness in capturing
both local and global context of the input data, making it suitable for detailed
analysis tasks.

Key Components:

1. Unet: This class implements the U-Net architecture, which features a symmetric
   structure with downsampling and upsampling paths connected by skip connections.
   The model efficiently combines local and global information, enabling precise
   localization in tasks such as image segmentation.

Highlights:

- The U-Net architecture in this implementation is enhanced with several advanced
  features, including:
  - Resnet Blocks: Incorporates residual learning concepts to ease the training of deep networks.
  - Attention Mechanisms: Employs both linear and traditional attention mechanisms to
    allow the model to focus on relevant parts of the input.
  - Time Embeddings: Optional incorporation of time embeddings for models that require
    temporal information processing, useful in generative tasks.

- The architecture is highly customizable, allowing adjustments in layer dimensions,
  number of channels, and depth, catering to a wide range of applications.

Usage:

This U-Net model can be integrated into various deep learning workflows, particularly
in TensorFlow-based environments. It can be adapted for a variety of tasks that require
detailed understanding and processing of spatial data, such as medical image analysis,
autonomous vehicle perception systems, and more.

The modularity of the architecture makes it a versatile tool in the deep learning
practitioner's toolkit, allowing for experimentation and adaptation to specific task
requirements.
"""

# Following this comment, the implementation of the Unet class and any supporting classes or functions would be placed.


import tensorflow.keras.layers as nn
import tensorflow_addons as tfa
from miniDiffusion.models.helpers import (
    default,
    SinusoidalPosEmb,
    GELU,
    Downsample,
    Residual,
    Identity,
    Upsample,
    PreNorm,
)
from miniDiffusion.models.blocks import ResnetBlock, LinearAttention, Attention
from tensorflow.keras import Model, Sequential
from functools import partial
import tensorflow as tf


class Unet(Model):
    def __init__(
        self,
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        sinusoidal_cond_mlp=True,
    ):
        super(Unet, self).__init__()

        # Initialization of model dimensions and configurations.
        self.channels = channels  # Number of input/output channels.
        init_dim = default(init_dim, dim // 3 * 2)  # Initial dimension for convolution.

        # Initial convolution layer with a kernel size of 7.
        self.init_conv = nn.Conv2D(
            filters=init_dim, kernel_size=7, strides=1, padding="SAME"
        )

        # Calculate the dimensions for each stage of the model based on multipliers.
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Partial function for creating Resnet blocks with specified groups.
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # Time embeddings configuration.
        time_dim = dim * 4
        self.sinusoidal_cond_mlp = sinusoidal_cond_mlp

        # MLP for time embeddings with sinusoidal positional embeddings and GELU activation.
        self.time_mlp = Sequential(
            [
                SinusoidalPosEmb(dim),
                nn.Dense(units=time_dim),
                GELU(),
                nn.Dense(units=time_dim),
            ],
            name="time embeddings",
        )

        # Layers for downsampling and upsampling in the U-Net architecture.
        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        # Downsample layers: consist of Resnet blocks, attention, and downsampling.
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                [
                    block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else Identity(),
                ]
            )

        # Middle layers of U-Net with Resnet blocks and attention.
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsample layers: similar to downsample layers, but in reverse order.
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                [
                    block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in) if not is_last else Identity(),
                ]
            )

        # Output dimension calculation, considering if variance is learned.
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        # Final convolutional layer for generating output.
        self.final_conv = Sequential(
            [
                block_klass(dim * 2, dim),
                nn.Conv2D(filters=self.out_dim, kernel_size=1, strides=1),
            ],
            name="output",
        )

    def call(self, x, time=None, training=True, **kwargs):
        # Initial convolution.
        x = self.init_conv(x)
        # Compute time embeddings.
        t = self.time_mlp(time)

        # List to store intermediate outputs for skip connections.
        h = []

        # Downsample path: apply blocks, attention, and store intermediate outputs.
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Apply mid blocks and attention.
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsample path: concatenate with stored outputs and apply blocks and attention.
        for block1, block2, attn, upsample in self.ups:
            x = tf.concat([x, h.pop()], axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        # Concatenate final output with the last stored output and apply final convolution.
        x = tf.concat([x, h.pop()], axis=-1)
        x = self.final_conv(x)
        return x
