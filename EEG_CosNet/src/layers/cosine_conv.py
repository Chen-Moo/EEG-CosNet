# -*- coding: utf-8 -*-
"""
Cosine Convolution Layer.

Implements a Keras custom layer that is functionally equivalent to a standard
2D convolution but replaces the inner-product operation with a **cosine
similarity**. The motivation is that the absolute amplitude of EEG signals
varies strongly across subjects and trials, while the discriminative
information (e.g. for depression recognition) is mainly carried by the
**shape and phase** of the waveform.

A standard convolution ``sum(x * w) + b`` scales its response linearly with
input amplitude, making the model sensitive to distribution shifts in the
amplitude scale. In contrast, cosine convolution L2-normalizes both the input
patch and the kernel before computing the inner product, compressing the
response into [-1, 1] and retaining only directional (i.e. waveform shape)
information, thereby gaining invariance to amplitude scaling.

Mathematical form::

    y = < x_hat, w_hat > = (x/||x||) . (w/||w||)

where ``x_hat`` and ``w_hat`` are the L2-normalized input patch and kernel,
respectively.
"""

import tensorflow as tf
from keras.layers import Layer


class CosineConv2D(Layer):
    """2D convolution layer whose operation is cosine similarity.

    This layer is API-compatible with ``keras.layers.Conv2D`` and can replace
    a standard convolution layer without modifying downstream dimension
    assumptions.

    Parameters
    ----------
    filters : int
        Number of output feature-map channels (number of kernels).
    kernel_size : int or tuple of int
        Spatial size of the kernel ``(kh, kw)``.
    strides : tuple of int, optional
        Strides, default ``(1, 1)``.
    padding : {'same', 'valid'}, optional
        Padding strategy, default ``'same'``.
    use_bias : bool, optional
        Whether to use a bias term. Not required in cosine convolution,
        default ``False``.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 use_bias=False,
                 **kwargs):
        super(CosineConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (tuple, list)) \
            else (kernel_size, kernel_size)
        self.strides = tuple(strides) if isinstance(strides, (tuple, list)) \
            else (strides, strides)
        self.padding = padding.upper()
        self.use_bias = use_bias

    def build(self, input_shape):
        """Construct trainable weights (kernel and optional bias) from the input shape."""
        kh, kw = self.kernel_size

        # ---- Kernel weights, shape (kh, kw, in_channels, filters) ----
        self.kernel = self.add_weight(
            name='kernel',
            shape=(kh, kw, int(input_shape[-1]), self.filters),
            initializer='glorot_uniform',
            trainable=True,
        )

        # ---- Optional bias ----
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer='zeros',
                trainable=True,
            )

        super(CosineConv2D, self).build(input_shape)

    def call(self, inputs):
        """Forward pass: extract patches -> L2 normalize -> cosine similarity -> restore map."""
        # 1) Extract sliding-window image patches (im2col)
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding,
        )

        batch_size = tf.shape(inputs)[0]
        output_height = tf.shape(patches)[1]
        output_width = tf.shape(patches)[2]

        # Flattened dimension of a single patch = kh * kw * in_channels
        patch_dim = self.kernel_size[0] * self.kernel_size[1] * inputs.shape[-1]

        # 2) Flatten patches: (B, H*W, patch_dim)
        patches = tf.reshape(
            patches, [batch_size, output_height * output_width, patch_dim])

        # 3) Flatten kernel: (patch_dim, filters)
        kernel = tf.reshape(self.kernel, [patch_dim, self.filters])

        # 4) L2-normalize input patches (along the last dimension)
        patches = tf.nn.l2_normalize(patches, axis=-1)

        # 5) L2-normalize kernels (along the first dimension, each filter independent)
        kernel = tf.nn.l2_normalize(kernel, axis=0)

        # 6) Cosine similarity (equivalent to inner product after normalization)
        output = tf.matmul(patches, kernel)

        # 7) Restore feature map (B, H, W, filters)
        output = tf.reshape(
            output, [batch_size, output_height, output_width, self.filters])

        # 8) Optional bias
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        return output

    def get_config(self):
        """Serialize configuration, supporting ``model.save`` / ``model.to_json``."""
        config = super(CosineConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
        })
        return config
