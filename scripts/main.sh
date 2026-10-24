# CARLA environment variables
export CARLA_VERSION="0915"
export CARLA_ROOT="${FMSR_PROJECT_ROOT}/3rd_party/CARLA_${CARLA_VERSION}"

# Prefer this checkout over any editable installation of another FMSR checkout.
export PYTHONPATH="${FMSR_PROJECT_ROOT}:${CARLA_ROOT}/PythonAPI/carla${PYTHONPATH:+:${PYTHONPATH}}"

# System paths
export PATH=$FMSR_PROJECT_ROOT:$PATH
export PATH=$FMSR_PROJECT_ROOT/scripts:$PATH

# NavSim
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${FMSR_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${FMSR_PROJECT_ROOT}/3rd_party/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="${FMSR_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1"
export OPENSCENE_DATA_ROOT="${FMSR_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset"

# Default Py123D Output Directory
export PY123D_DATA_ROOT="${FMSR_PROJECT_ROOT}/data/carla_leaderboard2_py123d"
