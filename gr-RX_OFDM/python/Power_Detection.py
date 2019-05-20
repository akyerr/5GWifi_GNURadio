#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2019 none.
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

import numpy as np
from gnuradio import gr


class Power_Detection(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(self,
                               name="Power_Detection",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        # Initialize Per Function Call
        x = np.zeros(1)
        count = 1

        for i in range(len(in0)):
            x = np.absolute(in0[i])**2 + x
            count += 1
        rms_power = np.sqrt(x/count)

        if rms_power > 0.05:
            out[:] = in0
            print('Signal!')
        else:
            out[:] = np.zeros(len(input_items[0]))
            print('No Signal!')

        return len(output_items[0])


