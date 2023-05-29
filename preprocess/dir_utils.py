
import os

def maybe_mkdir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)