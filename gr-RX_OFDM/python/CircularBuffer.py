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


class CircularBuffer(gr.sync_block):
    def __init__(self, buffer_size, no_zeros):
        gr.sync_block.__init__(self,
                               name="CircularBuffer",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])

        self.start_ptr = 0
        self.end_ptr = buffer_size - 1
        self.current_ptr = 0
        self.current_end_ptr = 0

        self.buffer_size = buffer_size
        self.data_buffer = np.zeros((1, buffer_size)) + 1j * np.zeros((1, buffer_size))
        self.no_zeros = no_zeros

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        if self.no_zeros == 1:
            in0 = in0[in0 != 0+0j]

        # Store to Buffer #
        self.end_ptr = (self.start_ptr + len(in0)) % self.buffer_size

        # print("Start PTR: ", self.start_ptr)
        # print("END PTR: ", self.end_ptr)
        if self.end_ptr < self.start_ptr:
            self.data_buffer[0, self.start_ptr: self.buffer_size] = \
                in0[0:(self.buffer_size - self.start_ptr)]
            self.data_buffer[0, 0: self.end_ptr] = in0[(self.buffer_size - self.start_ptr):(self.buffer_size - self.start_ptr + self.end_ptr)]
        else:
            self.data_buffer[0, self.start_ptr: self.end_ptr] = in0[:]
        self.start_ptr = self.end_ptr

        # Retrieve from Buffer #
        self.current_end_ptr = (self.current_ptr + len(output_items[0])) % self.buffer_size

        # print("Current PTR: ", self.current_ptr)
        # print("Current END PTR: ", self.current_end_ptr)
        output_buffer_size = len(in0)
        if self.current_end_ptr < self.current_ptr:
            out[0:(self.buffer_size - self.current_ptr)] = self.data_buffer[0, self.current_ptr: self.buffer_size]
            out[(self.buffer_size - self.current_ptr):(self.buffer_size - self.current_ptr + self.current_end_ptr)] = self.data_buffer[0, 0:self.current_end_ptr]
        else:
            out[:] = self.data_buffer[0, self.current_ptr: self.current_end_ptr]
        self.current_ptr = self.current_end_ptr
        # print("OUT: ", out[:])
        return len(output_items[0])

