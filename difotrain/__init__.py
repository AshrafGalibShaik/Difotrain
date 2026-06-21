"""DifoTrain: an embodiment-agnostic framework for training Vision-Language-Action
(VLA) models from real-world and simulated demonstrations.

The framework is organized around four pluggable interfaces:

* ``Robot``       (difotrain.embodiment) - sim or real hardware, one common API.
* ``DataSource``  (difotrain.data)       - teleop, human-video, sim rollouts.
* ``Policy``      (difotrain.policy)      - native or wrapped VLA models.
* ``EpisodeDataset`` (difotrain.data)     - standard, interoperable storage.

See ``difotrain.cli`` for the command line entry points.
"""

__version__ = "0.2.1"

__all__ = ["__version__"]
