"""Adapters that wrap external/pretrained VLA models behind :class:`Policy`.

The :class:`EchoVLAPolicy` is a dependency-free reference adapter showing the
contract a real OpenVLA / Octo / π0 / ACT wrapper must satisfy.
"""

from .echo import EchoVLAPolicy

__all__ = ["EchoVLAPolicy"]
