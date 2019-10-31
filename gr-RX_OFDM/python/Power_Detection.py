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
import pickle
import scipy.io
from gnuradio import gr


class Power_Detection(gr.sync_block):

    def __init__(self):
        gr.sync_block.__init__(self,
                               name="Power_Detection",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])
        self.window_size = 32
        self.window_slide_dist = 1
        self.mat_variable = dict()
        self.mat_variable['signal'] = np.zeros(1) + 1j

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        
        start = 0
        end = 0

        num_windows = int(np.ceil((len(in0) - self.window_size)/self.window_slide_dist) + 1)

        for i in range(num_windows):
            start = i * self.window_slide_dist
            if end > len(in0):
                end = len(in0)
            else:
                end = start + self.window_size
            x_win = in0[start:end]
            x_pow = np.sum(np.abs(x_win) ** 2) / self.window_size
            if x_pow > 0.01:
                # print("Signal Found!")
                # print("Good Signal... x_pow", x_pow)
                self.mat_variable['signal'] = in0
                print(self.mat_variable)
                scipy.io.savemat('/home/tayloreisman/Desktop/goodsignal.mat', self.mat_variable)
                out[:] = in0

            else:
                # print("Signal Not Found!")
                # print("Bad Signal... x_pow: ", x_pow)
                out[:] = in0
        return len(output_items[0])
