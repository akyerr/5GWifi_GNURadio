#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2019 gr-RX_OFDM author.
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
import scipy.io
from gnuradio import gr


class MAT_Sink(gr.sync_block):

    def __init__(self, file_name, directory, variable_name):
        gr.sync_block.__init__(self,
                               name="MAT_Sink",
                               in_sig=[np.complex64],
                               out_sig=None)

        self.directory = directory
        self.file_name = file_name
        self.path = str(self.directory + file_name)
        self.mat_variable = dict()
        self.matlab_variable_name = str(variable_name)
        self.mat_variable['iq_data'] = np.zeros(1) + 1j

    def work(self, input_items, output_items):
        in0 = input_items[0]

        self.mat_variable['iq_data'] = in0
        # print(self.mat_variable)
        scipy.io.savemat(self.path, self.mat_variable)

        return len(input_items[0])

