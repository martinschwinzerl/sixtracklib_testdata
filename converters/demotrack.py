#!/usr/bin/env python
# -*- coding: utf-8 -*-

import struct
import numpy as np

def float_to_bytes( value, format_str="<d", dtype=np.float64 ):
    return bytes( struct.pack( format_str, dtype( value ) ) )
