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
from gnuradio import gr

class Frequency_Offset_Reporter(gr.sync_block):
    """
    docstring for block Frequency_Offset_Reporter
    """
    def __init__(self):
        gr.sync_block.__init__(self,
                               name="Frequency_Offset_Reporter",
                               in_sig=[np.complex64],
                               out_sig=None)
        self.quad1 = np.array([np.sqrt(2) / 2 + (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.quad2 = np.array([-np.sqrt(2) / 2 + (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.quad3 = np.array([-np.sqrt(2) / 2 - (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.quad4 = np.array([np.sqrt(2) / 2 - (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.mag_min_buffer = np.zeros([1])
        self.work_call = 0
        self.clear_condition = 0

    def work(self, input_items, output_items):
        in0 = input_items[0]

        for test in in0:
            if test != 0j:
                temp = min(np.absolute(self.quad1 - test), np.absolute(self.quad2 - test),
                           np.absolute(self.quad3 - test), np.absolute(self.quad4 - test))
                mag_min = np.abs(temp)
                self.mag_min_buffer = np.append(self.mag_min_buffer, mag_min)
        if self.clear_condition is 0:
            self.mag_min_buffer = np.delete(self.mag_min_buffer, 0)
            self.clear_condition = 1
        mag_min_mu = np.average(self.mag_min_buffer)
        mag_min_sigma = np.std(self.mag_min_buffer)

        print("Average Distance of IQ Points: ", mag_min_mu)
        print("Average Deviation of IQ Points: ", mag_min_sigma)
        if self.work_call % 50 == 0:
            self.mag_min_buffer = np.zeros([1])
            self.clear_condition = 0
        self.work_call += 1

        return len(input_items[0])

