import os
import re
from typing import Dict, Optional, List, Union, Callable, Any, Tuple
import time, datetime
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import jax, optax
from jax import random, numpy as jnp
import jax.lax as lax
import numpy as np

from flax import linen as nn
from flax import jax_utils
from flax.training import train_state, checkpoints
from flax.training import common_utils

from config import Config

class PositionalEncoder(nn.Module):
  config: Config
  """Adds sinusoidal positional embeddings to the inputs.
     
  Attribues:
     config: config class containing hyperparameters.
  """
  @nn.compact
  def __call__(self, 
               x: jnp.array
               ) -> jnp.array:
    """Applys PositionalEncoder Module.
     
    Args:
      x: inputs of shape [batch_size, length, features].

    Returns: 
      The positional embedded inputs of the same shape of the inputs.
    """

    assert x.ndim == 3                                                          # [Batch, SeqLen, EmbedDim]
    config = self.config

    batch_size, seq_len = x.shape[0], x.shape[1]
    pe = jnp.empty((batch_size, seq_len, config.embed_dim))                     # [Batch, SeqLen, EmbedDim]
    position = jnp.arange(0, seq_len, dtype=config.dtype)[jnp.newaxis, :]       # [1, SeqLen]
    div_term = jnp.exp(jnp.arange(0, config.embed_dim, 2, dtype=config.dtype)
         * (-lax.log(10000.0) / config.embed_dim))[jnp.newaxis, :]              # [1, EmbedDim/2]
    radians = jnp.einsum('ij,kl->jl', position, div_term)                       # [SeqLen, EmbedDim/2]
    pe = pe.at[:, :, 0::2].set(jnp.sin(radians))
    pe = pe.at[:, :, 1::2].set(jnp.cos(radians))
    x = x + pe
    return x.astype(config.dtype)
  

class MultiHeadAttention(nn.Module):
  config: Config
  """Multi-head dot-product attention.
     
  Attribues:
     config: config class containing hyperparameters.
  """

  @nn.compact
  def __call__(self, 
               q : jnp.ndarray,
               k : jnp.ndarray,
               v : jnp.ndarray, 
               mask : jnp.ndarray = None
               ) -> Tuple[jnp.array, jnp.array]:
    """Applys MultiHeadAttention Module.
     
    Args:
      q: query inputs of shape [batch_size, query_length, features].
      k: key inputs of shape [batch_size, key/value_length, features].
      v: value inputs of shape [batch_size, key/value_length, features].
      mask: attention mask of shape [batch_size, 1, query_length, key/value_length].
       
    Returns: 
      The output of shape [batch_size, query_length, features], 
      and an attention matrix of shape [batch_size, num_heads, query_length, key/value_length].
    """
    
    assert q.ndim == 3                                                          # [Batch, SeqLen_q, EmbedDim]
    assert k.ndim == 3                                                          # [Batch, SeqLen_k, EmbedDim]
    assert v.ndim == 3                                                          # [Batch, SeqLen_v, EmbedDim]
    assert q.shape[0] == k.shape[0] == v.shape[0]                               # Same batch size
    assert k.shape[1] == v.shape[1]                                             # SeqLen_k = SeqLen_v
    if mask is not None:
      assert mask.ndim == 4                                                     # [Batch, 1, SeqLen_q, SeqLen_k]
    config = self.config

    q_seq_len, k_seq_len = q.shape[1], k.shape[1]

    q = nn.Dense(config.num_heads * config.k_dim)(q)                            # [Batch, SeqLen_q, Head * Dim_k]
    k = nn.Dense(config.num_heads * config.k_dim)(k)                            # [Batch, SeqLen_k, Head * Dim_k]
    v = nn.Dense(config.num_heads * config.v_dim)(v)                            # [Batch, SeqLen_k, Head * Dim_v]

    q = q.reshape(-1, q_seq_len, config.num_heads, config.k_dim)                # [Batch, SeqLen_q, Head, Dim_k]
    k = k.reshape(-1, k_seq_len, config.num_heads, config.k_dim)                # [Batch, SeqLen_k, Head, Dim_k]
    v = v.reshape(-1, k_seq_len, config.num_heads, config.v_dim)                # [Batch, SeqLen_k, Head, Dim_v]

    attention = (jnp.einsum('...qhd,...khd->...hqk', q, k) 
                                / jnp.sqrt(config.v_dim)).astype(config.dtype)  # [Batch, Head, SeqLen_q, SeqLen_k]
    if mask is not None:                                                         
      attention = jnp.where(mask, attention, -jnp.inf)                          # Change the masked position to -jnp.inf.
    attention = nn.softmax(attention, axis=-1).astype(config.dtype)             # [Batch, Head, SeqLen_q, SeqLen_k]
    values = jnp.einsum('...hqk,...khd->...qhd', attention, v)                  # [Batch, SeqLen_q, Head, Dim_v]
    values = values.reshape(-1, q_seq_len, config.num_heads * config.v_dim)     # [Batch, SeqLen_q, Head × Dim_v (=EmbedDim)]
    out = nn.Dense(config.embed_dim, dtype=config.dtype)(values)                # [Batch, SeqLen_q, EmbedDim]
    return out.astype(config.dtype), attention.astype(config.dtype)
  
class FeedForward(nn.Module):
  config: Config
  """Feed Forward Network.
     
  Attribues:
     config: config class containing hyperparameters.
  """

  @nn.compact
  def __call__(self, 
               x: jnp.array,
               deterministic: bool
               ) -> jnp.array:
    """Applys FeedForward module.
     
    Args:
      x: inputs of shape [batch_size, length, features].
      deterministic: parameter for nn.Dropout. if true, it masks and scales the inputs. during training, it should be False. Otherwise True.

    Returns: 
      The output of the same shape of the inputs.
    """
    
    assert x.ndim == 3                                                          # [Batch, SeqLen, EmbedDim]
    config = self.config

    x = nn.Dense(config.ff_dim, dtype=config.dtype)(x)                          # Dense Layer
    x = nn.relu(x)                                                              # ReLu
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)    # Dropout
    x = nn.Dense(config.embed_dim, dtype=config.dtype)(x)                       # Dense Layer
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)    # Dropout
    return x.astype(config.dtype)

class TransformerDecoderLayer(nn.Module):
  config: Config
  """Transformer decoder layer (Encoder Decoder 1D block).
     
  Attribues:
     config: config class containing hyperparameters.
  """

  @nn.compact
  def __call__(self,
              x: jnp.array,
              memory: jnp.array,
              decoder_mask: jnp.array,
              encoder_decoder_mask: jnp.array,
              deterministic: bool
              )-> Tuple[jnp.array, jnp.array, jnp.array]:
    """Applys TransformerDecoderLayer module.
     
    Args:
      x: inputs of shape [batch_size, x_length, features].
      memory: encoded sources from Transformer Encoder of shape [batch_size, memory_length, features].
      decoder_mask: attention mask for Self-Attention of shape [batch_size, 1, x_length, x_length].
      encoder_decoder_mask: attention mask for Source-Target Attention of shape [batch_size, 1, x_length, memory_length].
      deterministic: parameter for nn.Dropout. if true, it mask and scale the inputs. during training, it should be False. Otherwise True.

    Returns: 
      The outputs of the same shape of the inputs,
      an attention matrix in Self-Attention of shape [batch_size, num_heads, x_length, x_length],
      and an attention matrix in Source-Target Attention of shape [batch_size, num_heads, x_length, memory_length].
    """
    
    assert x.ndim == 3                                                          # [Batch, SeqLen, EmbedDim]                         
    assert memory.ndim == 3                                                     # [Batch, SeqLen, EmbedDim]
    assert decoder_mask.ndim == 4                                               # [Batch, 1, SeqLen_q, SeqLen_k]
    assert encoder_decoder_mask.ndim == 4                                       # [Batch, 1, SeqLen_q, SeqLen_k]
    config = self.config

    res = x                                                                     # Residual
    x, self_attention = MultiHeadAttention(config)(x, x, x,
                                                   mask=decoder_mask)           # Self-Attention
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)    # Dropout
    x = nn.LayerNorm(dtype=config.dtype)(res + x)                               # Add & Norm

    res = x                                                                     # Residual
    x, src_trg_attention = MultiHeadAttention(config)(x, memory, memory,
                                                      mask=encoder_decoder_mask)# Source-Target Attention
    x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=deterministic)    # Dropout
    x = nn.LayerNorm(dtype=config.dtype)(res + x)                               # Add & Norm

    res = x                                                                     # Residual
    x = FeedForward(config)(x, deterministic)                                   # Feed Forward Network
    x = nn.LayerNorm(dtype=config.dtype)(res + x)                               # Add & Norm

    return x.astype(config.dtype), self_attention.astype(config.dtype), src_trg_attention.astype(config.dtype)

class GPT2(nn.Module):
  config: Config
  """Generative Pretrained Transformer (GPT-2).

  Attribues:
     config: config class containing hyperparameters.
  """

  @nn.compact
  def __call__(self,
             src: jnp.array,
             train: bool = False,
             return_attn: bool = False
             ) -> Union[jnp.array, Tuple[jnp.array, List[jnp.array]]]:
    """Applys GPT-2 module.

    Args:
      src: targets of shape [batch_size, src_length].
      train: For training, it should be True. Otherwise False.
      return_attn: if true, returns list of Self-Attention matrixes for the number of layers.

    Returns:
      If return_attn is True,
        the logits of shape [batch_size, src_length, target_vocab_size],
        and list of attention matrixes in Self-Attention for the number of layers,
      else,
        the logits.
    """

    assert src.ndim == 2                                                        # [Batch, SeqLen]
    config = self.config

    decoder_mask = nn.combine_masks(
        nn.make_attention_mask(
            jnp.ones_like(src),
            src != config.pad_id,
            dtype=bool),
        nn.make_causal_mask(src,
                            dtype=bool)
    )

    x = nn.Embed(num_embeddings=config.vocab_size,
                 features=config.embed_dim,
                 dtype=config.id_dtype,
                 param_dtype=config.dtype,
                 embedding_init=nn.initializers.normal(stddev=1.0))(src)        # Embedding
    x = PositionalEncoder(config)(x)                                            # Positinal Encoding
    # x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=not train)        # Dropout

    x = x.astype(config.dtype)
    self_attention_list = []
    for layer in [TransformerDecoderLayer(config) for _ in range(config.num_layers)]:
      x, self_attention = layer(x, decoder_mask, not train)                     # Decoder Layer
      self_attention_list.append(self_attention)
    x = nn.LayerNorm(dtype=config.dtype)(x)                                     # Norm berfore logits
    logits = nn.Dense(config.vocab_size, dtype=config.dtype)(x)                 # Dense Layer to reform the embed size to vocab size

    if return_attn:
      return logits.astype(config.dtype), self_attention_list
    return logits.astype(config.dtype)                                          # [Batch, SeqLen, VocabSize]