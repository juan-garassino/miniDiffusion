"""
Neural Network Layers for Deep Learning Models

This Python file includes custom neural network layers tailored for use in deep learning
applications. Implemented using TensorFlow, a powerful open-source machine learning
library, these layers are designed to be integrated into various neural network
architectures, particularly for tasks like image processing, sequence modeling, and
generative models.

Classes:

1. Block: A fundamental neural network building block combining convolutional operations,
   group normalization, and the SiLU (Sigmoid Linear Unit) activation function. This layer
   is versatile and can be used in different architectures, especially in convolutional neural
   networks (CNNs).

2. ResnetBlock: An extension of the Block class, incorporating the principles of residual
   learning. It is suited for deep networks, using skip connections to facilitate training.
   The optional use of time embeddings makes this block applicable in models that require
   encoding of temporal information, like generative models.

3. LinearAttention: Implements a linear attention mechanism, an efficient alternative to
   the standard attention mechanism. This layer is designed to reduce computational
   demands while maintaining the ability to focus on specific input parts. Ideal for models
   where traditional attention mechanisms are computationally expensive.

4. Attention: A classic attention layer based on scaled dot-product attention. It enables
   the model to focus on various input sequence parts, essential in models that need to
   process contextual or relational information, such as transformers.

Usage:

Each class is designed to be modular, allowing for easy integration into larger models.
They are built on TensorFlow's Layer class, ensuring compatibility with the TensorFlow
ecosystem and facilitating their use within custom model architectures.

This file serves as a foundational component for constructing sophisticated neural network
models across a diverse range of machine learning and artificial intelligence applications.
"""

# Imports

from miniDiffusion.model.custom_layers.layers import SiLU, exists, Identity
import tensorflow.keras.layers as nn
import tensorflow_addons as tfa
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tensorflow import einsum
from einops import rearrange
import tensorflow as tf


class Block(Layer):
    def __init__(self, dim, groups=8):
        super(Block, self).__init__()
        # Initialize a 2D convolutional layer with 'dim' filters, a kernel size of 3,
        # stride of 1, and 'SAME' padding to keep input and output dimensions the same.
        self.proj = nn.Conv2D(dim, kernel_size=3, strides=1, padding="SAME")

        # Group normalization layer with 'groups' number of groups, providing stability
        # to the learning process by normalizing across grouped feature channels.
        self.norm = tfa.layers.GroupNormalization(groups, epsilon=1e-05)

        # Activation function SiLU (Sigmoid Linear Unit), which can help the network
        # learn complex patterns and generally performs better than traditional functions
        # like ReLU.
        self.act = SiLU()

    def call(self, x, gamma_beta=None, training=True):
        # Apply the convolutional projection to the input.
        x = self.proj(x)

        # Apply group normalization. 'training' flag helps in distinguishing
        # between training and inference phases.
        x = self.norm(x, training=training)

        # If gamma and beta parameters are provided (style-based modulation),
        # apply them to modulate the normalization output.
        if exists(gamma_beta):
            gamma, beta = gamma_beta
            x = x * (gamma + 1) + beta

        # Apply the SiLU activation function to the transformed input.
        x = self.act(x)
        return x


class ResnetBlock(Layer):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super(ResnetBlock, self).__init__()

        # A multi-layer perceptron (MLP) that is only initialized if a time embedding
        # dimension is provided. This is typically used in models that need to encode
        # temporal information, like in generative models conditioned on time.
        self.mlp = (
            Sequential([SiLU(), nn.Dense(units=dim_out * 2)])
            if exists(time_emb_dim)
            else None
        )

        # Two blocks that apply convolution, normalization, and activation.
        # These are the main computational units of the ResnetBlock.
        self.block1 = Block(dim_out, groups=groups)
        self.block2 = Block(dim_out, groups=groups)

        # A convolutional layer with a kernel size of 1 (pointwise convolution) used
        # for matching the dimensions if the input and output dimensions differ.
        # If dimensions are the same, an identity function is used instead.
        self.res_conv = (
            nn.Conv2D(filters=dim_out, kernel_size=1, strides=1)
            if dim != dim_out
            else Identity()
        )

    def call(self, x, time_emb=None, training=True):
        gamma_beta = None
        # If both MLP and time embedding are provided, process the time embedding
        # through the MLP and prepare gamma and beta for conditional normalization.
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            # Rearrange the time embedding to match the spatial dimensions of the feature map.
            time_emb = rearrange(time_emb, "b c -> b 1 1 c")
            # Split the time embedding into two parts for gamma and beta.
            gamma_beta = tf.split(time_emb, num_or_size_splits=2, axis=-1)

        # Apply the first block with conditional normalization if gamma_beta is available.
        h = self.block1(x, gamma_beta=gamma_beta, training=training)

        # Apply the second block.
        h = self.block2(h, training=training)

        # Return the result of the residual connection, adding the transformed input
        # to the original input after possibly adjusting its dimensions.
        return h + self.res_conv(x)


class LinearAttention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        # Scaling factor to normalize the query vectors.
        self.scale = dim_head**-0.5
        # The number of attention heads.
        self.heads = heads
        # Total dimension of all heads combined.
        self.hidden_dim = dim_head * heads

        # Softmax layer for attention computation.
        self.attend = nn.Softmax()
        # A convolutional layer that transforms input feature maps into
        # query, key, and value representations for attention mechanism.
        self.to_qkv = nn.Conv2D(
            filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False
        )

        # Sequential model that projects the output of the attention layer back to the
        # original input dimension and applies layer normalization.
        self.to_out = Sequential(
            [
                nn.Conv2D(filters=dim, kernel_size=1, strides=1),
                nn.LayerNormalization(axis=3),
            ]
        )

    def call(self, x, training=True):
        b, h, w, c = x.shape  # Batch size, height, width, channels
        # Apply convolutional layer to generate query, key, and value.
        qkv = self.to_qkv(x)
        # Split the output into query, key, and value components.
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        # Reshape and distribute each of q, k, v across heads.
        q, k, v = map(
            lambda t: rearrange(t, "b x y (h c) -> b h c (x y)", h=self.heads), qkv
        )

        # Apply softmax to queries and keys for normalization.
        q = tf.nn.softmax(q, axis=-2)
        k = tf.nn.softmax(k, axis=-1)

        # Scale query vectors.
        q = q * self.scale
        # Compute the context as a weighted sum of values.
        context = einsum("b h d n, b h e n -> b h d e", k, v)

        # Output attention by combining context with query.
        out = einsum("b h d e, b h d n -> b h e n", context, q)
        # Rearrange output to match the shape of the input feature map.
        out = rearrange(out, "b h c (x y) -> b x y (h c)", h=self.heads, x=h, y=w)
        # Apply the final convolutional and normalization layers.
        out = self.to_out(out, training=training)

        return out


class Attention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        # Scaling factor for the query vectors to avoid large values during dot products.
        self.scale = dim_head**-0.5
        # Number of attention heads. Multiple heads allow the model to jointly attend to
        # information from different representation subspaces.
        self.heads = heads
        # Combined dimension of all heads.
        self.hidden_dim = dim_head * heads

        # Convolutional layer to generate query (q), key (k), and value (v) vectors.
        self.to_qkv = nn.Conv2D(
            filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False
        )
        # Convolutional layer to project the output of the attention back to the original
        # dimension of the input.
        self.to_out = nn.Conv2D(filters=dim, kernel_size=1, strides=1)

    def call(self, x, training=True):
        b, h, w, c = x.shape  # Batch size, height, width, channels
        # Generate query, key, and value vectors.
        qkv = self.to_qkv(x)
        # Split the result into individual q, k, and v components.
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        # Reshape and distribute each of q, k, v across heads.
        q, k, v = map(
            lambda t: rearrange(t, "b x y (h c) -> b h c (x y)", h=self.heads), qkv
        )
        # Scale the query vectors.
        q = q * self.scale

        # Compute similarity scores between queries and keys.
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        # Find the maximum of each similarity vector and use it for numerical stability.
        sim_max = tf.stop_gradient(tf.expand_dims(tf.argmax(sim, axis=-1), axis=-1))
        sim_max = tf.cast(sim_max, tf.float32)
        # Normalize similarity scores.
        sim = sim - sim_max
        # Apply softmax to get attention weights.
        attn = tf.nn.softmax(sim, axis=-1)

        # Compute weighted sum of values based on attention weights.
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        # Rearrange the output to match the shape of the input feature map.
        out = rearrange(out, "b h (x y) d -> b x y (h d)", x=h, y=w)
        # Project the attention output back to the original dimension.
        out = self.to_out(out, training=training)

        return out
