import torch
import numpy as np
import os
import elevation

a = elevation.clip(bounds=(12.35, 41.8, 12.65, 42))
print(a)

elevation.clean()