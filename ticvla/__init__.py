"""TIC-VLA public package."""

__all__ = ["TICVLA"]


def __getattr__(name):
    if name == "TICVLA":
        from ticvla.models.ticvla import TICVLA
        return TICVLA
    raise AttributeError(name)
