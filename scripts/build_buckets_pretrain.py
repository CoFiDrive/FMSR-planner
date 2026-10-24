from fmsr.data_loader.carla_dataset import CARLAData
from fmsr.training.config_training import TrainingConfig

config = TrainingConfig()
config.force_rebuild_bucket = True
config.use_planning_decoder = False

data = CARLAData(
    root=config.carla_data,
    config=config,
    training_session_cache=None,
    build_cache=False
)
