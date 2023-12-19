import dataclasses

@dataclasses.dataclass
class Config:
  # Architecture Config
  vocab_size: int = 8000
  unk_id: int = 0
  bos_id: int = 1
  eos_id: int = 2
  pad_id: int = 3
  max_len: int = 256
  embed_dim: int = 256
  id_dtype: Any = jnp.int16
  dtype: Any = jnp.float32
  num_heads: int = 32
  num_layers: int = 14
  q_dim: int = embed_dim // num_heads
  v_dim: int = embed_dim // num_heads
  ff_dim: int = 512
  dropout_rate: float = 0.10

  # Training Config
  special_ids: List[int] = [0,1,2,3]
  special_tokens: List[str] = ['<unk>','<s>', '</s>', '<pad>']
  seed: int = 42
  batch_size: int = 256
  learning_rate: float = 0.010                                                  #変更
  warmup_steps: int = 500
  num_epochs: int = 20
  save_ckpt_every_epochs: int = 1
  restore_checkpoints: bool = True
  ckpt_prefix: str = 'completion_ckpt_'
  ckpt_dir: str = '/content/drive/My Drive/checkpoints/completion/'
  spm_model: str = '/content/drive/MyDrive/wiki40b_ja/wiki40b_ja_test.model'
  train_text: str = '/content/drive/MyDrive/wiki40b_ja/wiki40b_ja_test.txt'