import tensorflow as tf
from tensorflow.keras import layers

from miniDiffusion.models.helpers import SiLU, exists, Identity
import tensorflow.keras.layers as nn
import tensorflow_addons as tfa
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tensorflow import einsum
from einops import rearrange
import tensorflow as tf

class Block3D(layers.Layer):
    def __init__(self, dim, groups=8):
        super(Block3D, self).__init__()
        # Initialize a 3D convolutional layer with 'dim' filters, a kernel size of 3,
        # stride of 1, and 'SAME' padding to keep input and output dimensions the same.
        self.proj = layers.Conv3D(dim, kernel_size=3, strides=1, padding="same")

        # Layer normalization layer for stability during training
        self.norm = layers.LayerNormalization(epsilon=1e-05)

        # Activation function SiLU (Sigmoid Linear Unit)
        self.act = tf.keras.activations.selu

    def call(self, x, gamma_beta=None, training=True):
        # Apply the convolutional projection to the input.
        x = self.proj(x)

        # Apply layer normalization. 'training' flag helps in distinguishing
        # between training and inference phases.
        x = self.norm(x, training=training)

        # If gamma and beta parameters are provided (style-based modulation),
        # apply them to modulate the normalization output.
        if gamma_beta is not None:
            gamma, beta = gamma_beta
            x = x * (gamma + 1) + beta

        # Apply the SiLU activation function to the transformed input.
        x = self.act(x)
        return x

class ResnetBlock(layers.Layer):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super(ResnetBlock, self).__init__()

        # A multi-layer perceptron (MLP) that is only initialized if a time embedding
        # dimension is provided. This is typically used in models that need to encode
        # temporal information, like in generative models conditioned on time.
        self.mlp = (
            Sequential([SiLU(), nn.Dense(units=dim_out * 2)])
            if time_emb_dim is not None
            else None
        )

        # Two blocks that apply convolution, normalization, and activation.
        # These are the main computational units of the ResnetBlock.
        self.block1 = Block3D(dim_out, groups=groups)
        self.block2 = Block3D(dim_out, groups=groups)

        # A convolutional layer with a kernel size of 1 (pointwise convolution) used
        # for matching the dimensions if the input and output dimensions differ.
        # If dimensions are the same, an identity function is used instead.
        self.res_conv = (
            layers.Conv3D(filters=dim_out, kernel_size=1, strides=1)
            if dim != dim_out
            else layers.Lambda(lambda x: x)
        )

    def call(self, x, time_emb=None, training=True):
        gamma_beta = None
        # If both MLP and time embedding are provided, process the time embedding
        # through the MLP and prepare gamma and beta for conditional normalization.
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            # Rearrange the time embedding to match the spatial dimensions of the feature map.
            time_emb = tf.expand_dims(tf.expand_dims(time_emb, axis=1), axis=1)
            # Split the time embedding into two parts for gamma and beta.
            gamma_beta = tf.split(time_emb, num_or_size_splits=2, axis=-1)

        # Apply the first block with conditional normalization if gamma_beta is available.
        h = self.block1(x, gamma_beta=gamma_beta, training=training)

        # Apply the second block.
        h = self.block2(h, training=training)

        # Return the result of the residual connection, adding the transformed input
        # to the original input after possibly adjusting its dimensions.
        return h + self.res_conv(x)

class LinearAttention(layers.Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        # Scaling factor to normalize the query vectors.
        self.scale = dim_head ** -0.5
        # The number of attention heads.
        self.heads = heads
        # Total dimension of all heads combined.
        self.hidden_dim = dim_head * heads

        # Softmax layer for attention computation.
        self.attend = tf.keras.layers.Softmax()
        # A convolutional layer that transforms input feature maps into
        # query, key, and value representations for attention mechanism.
        self.to_qkv = layers.Conv3D(
            filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False
        )

        # Sequential model that projects the output of the attention layer back to the
        # original input dimension and applies layer normalization.
        self.to_out = tf.keras.Sequential(
            [
                layers.Conv3D(filters=dim, kernel_size=1, strides=1),
                layers.LayerNormalization(axis=-1),
            ]
        )

    def call(self, x, training=True):
        b, d, h, w, c = x.shape  # Batch size, depth, height, width, channels
        # Apply convolutional layer to generate query, key, and value.
        qkv = self.to_qkv(x)
        # Split the output into query, key, and value components.
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        # Reshape and distribute each of q, k, v across heads.
        q, k, v = map(
            lambda t: tf.reshape(t, [b, self.heads, d * h * w, self.hidden_dim]),
            qkv
        )

        # Apply softmax to queries and keys for normalization.
        q = self.attend(q)
        k = self.attend(k)

        # Scale query vectors.
        q = q * self.scale
        # Compute the context as a weighted sum of values.
        context = tf.einsum("bhdc, bhde -> bhce", k, v)

        # Output attention by combining context with query.
        out = tf.einsum("bhde, bhcd -> bhce", context, q)
        # Rearrange output to match the shape of the input feature map.
        out = tf.reshape(out, [b, d, h, w, self.hidden_dim])
        # Apply the final convolutional and normalization layers.
        out = self.to_out(out, training=training)

        return out

class Attention(layers.Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        # Scaling factor for the query vectors to avoid large values during dot products.
        self.scale = dim_head ** -0.5
        # Number of attention heads. Multiple heads allow the model to jointly attend to
        # information from different representation subspaces.
        self.heads = heads
        # Combined dimension of all heads.
        self.hidden_dim = dim_head * heads

        # Convolutional layer to generate query (q), key (k), and value (v) vectors.
        self.to_qkv = layers.Conv3D(
            filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False
        )
        # Convolutional layer to project the output of the attention back to the original
        # dimension of the input.
        self.to_out = layers.Conv3D(filters=dim, kernel_size=1, strides=1)

    def call(self, x, training=True):
        b, d, h, w, c = x.shape  # Batch size, depth, height, width, channels
        # Generate query, key, and value vectors.
        qkv = self.to_qkv(x)
        # Split the result into individual q, k, and v components.
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        # Reshape and distribute each of q, k, v across heads.
        q, k, v = map(
            lambda t: tf.reshape(t, [b, self.heads, d * h * w, self.hidden_dim]),
            qkv
        )
        # Scale the query vectors.
        q = q * self.scale

        # Compute similarity scores between queries and keys.
        sim = tf.einsum("bhdi,bhdj->bhij", q, k)
        # Apply softmax to get attention weights.
        attn = tf.nn.softmax(sim, axis=-1)

        # Compute weighted sum of values based on attention weights.
        out = tf.einsum("bhij,bhdj->bhid", attn, v)
        # Reshape the output to match the shape of the input feature map.
        out = tf.reshape(out, [b, d, h, w, self.hidden_dim])
        # Project the attention output back to the original dimension.
        out = self.to_out(out, training=training)

        return out
