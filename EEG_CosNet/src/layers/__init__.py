# -*- coding: utf-8 -*-
"""
EEG-CosNet custom layers module.

This package contains two core custom computation units that constitute the
main technical contributions of this work:

1. :class:`CosineConv2D`
   -- Cosine convolution layer. It replaces the inner-product-plus-bias
   operation of a standard convolution with an L2-normalized cosine-similarity
   operation, so that the convolution response is determined solely by feature
   direction and is insensitive to amplitude scaling. This better matches the
   characteristics of EEG signals, whose amplitude fluctuates strongly while
   waveform shape and phase carry the discriminative information.

2. :class:`SpatialDropoutAttention`
   -- Spatial dropout + multi-head self-attention (MHSA) composite module.
   Inserted after each depthwise convolution block, it first applies
   structured random deactivation (spatial dropout) to the feature map to
   suppress overfitting in the channel/spatial dimensions, and then models
   long-range dependencies via multi-head self-attention, notably improving
   training stability and convergence.
"""

from src.layers.cosine_conv import CosineConv2D
from src.layers.attention import SpatialDropoutAttention

__all__ = [
    "CosineConv2D",
    "SpatialDropoutAttention",
]
