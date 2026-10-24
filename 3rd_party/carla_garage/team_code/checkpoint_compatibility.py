"""Local checkpoint initialization without redundant ImageNet downloads."""


def configure_local_checkpoint_initialization(config, load_file):
  if getattr(config, 'use_fmsr', False) and load_file:
    # The TF++ checkpoint supplies the frozen image encoder, including BN.
    config.image_encoder_pretrained = False


def validate_image_encoder_checkpoint(model_state, checkpoint_state):
  prefix = 'backbone.image_encoder.'
  required = [key for key in model_state
              if key.startswith(prefix) and not key.endswith('.num_batches_tracked')]
  if not required:
    raise RuntimeError('Expected image encoder weights in the FMSR model')
  missing = [key for key in required if key not in checkpoint_state]
  mismatch = [key for key in required if key in checkpoint_state and
              checkpoint_state[key].shape != model_state[key].shape]
  if missing or mismatch:
    raise RuntimeError('Local TF++ checkpoint must fully initialize the image encoder: '
                       f'missing={missing[:5]}, shape_mismatch={mismatch[:5]}')
  return len(required)
