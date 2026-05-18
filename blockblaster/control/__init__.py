"""blockblaster.control — platform-agnostic device abstraction and auto-player.

Public surface::

    from blockblaster.control import make_device, Device
    from blockblaster.control.auto_player import run as auto_play
"""

from blockblaster.control.device import Device, InputNotSupportedError, make_device

__all__ = ["Device", "InputNotSupportedError", "make_device"]
