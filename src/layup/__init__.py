import os

# create symlinks for the assist and rebound shared objects
# in the main directory, ensuring that they are discoverable
# to cpython at runtime.
if not os.path.isfile("./libassist.so"):
    os.symlink("./include/assist/src/libassist.so", "./libassist.so")

if not os.path.isfile("./librebound.so"):
    os.symlink("./include/rebound/src/librebound.so", "./librebound.so")


from .example_module import greetings, meaning

__all__ = ["greetings", "meaning"]
