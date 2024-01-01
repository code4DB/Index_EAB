import os

from swirl_selection.stable_baselines.a2c import A2C
from swirl_selection.stable_baselines.acer import ACER
from swirl_selection.stable_baselines.acktr import ACKTR
from swirl_selection.stable_baselines.deepq import DQN
from swirl_selection.stable_baselines.her import HER
from swirl_selection.stable_baselines.ppo2 import PPO2
from swirl_selection.stable_baselines.td3 import TD3
from swirl_selection.stable_baselines.sac import SAC

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from ddpg import DDPG
    from gail import GAIL
    from ppo1 import PPO1
    from trpo_mpi import TRPO
del mpi4py

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
with open(version_file, 'r') as file_handler:
    __version__ = file_handler.read().strip()
