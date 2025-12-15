# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
try:
    from .models import *
except Exception:  # optional for hub usage
    pass
try:
    from .cait_models import *
except Exception:
    pass
try:
    from .resmlp_models import *
except Exception:
    pass
#from patchconvnet_models import *

dependencies = ["torch", "torchvision", "timm"]
