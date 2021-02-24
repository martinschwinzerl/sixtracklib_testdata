#!/usr/bin/env python
# -*- coding: utf-8 -*-

import struct
import numpy as np

def f64_to_bytes( value, endianess="<" ):
    return bytes( struct.pack( f"{endianess}d", np.float64( value ) ) )

