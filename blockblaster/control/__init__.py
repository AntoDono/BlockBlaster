"""blockblaster.control — platform-agnostic device abstraction."""

from blockblaster.control.device import Device, InputNotSupportedError, make_device

__all__ = ["Device", "InputNotSupportedError", "make_device"]
