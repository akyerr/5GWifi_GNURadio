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


class FrequencyOffsetCorrection(gr.sync_block):
    """
    docstring for block FrequencyOffsetCorrection
    """
    def __init__(self, min, max, step, sample_frequency):
        gr.sync_block.__init__(self,
                               name="Frequency Offset Correction",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])
        self.min = min
        self.max = max
        self.step_size = step
        self.span = self.max - self.min
        self.num_steps = int(np.ceil(self.span/self.step_size))
        self.sample_frequency = sample_frequency
        self.sample_period = 1/self.sample_frequency

        self.freq_buffer = range(self.min, self.max, self.step_size)

        # Target Points
        self.quad1 = np.array([np.sqrt(2) / 2 + (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.quad2 = np.array([-np.sqrt(2) / 2 + (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.quad3 = np.array([-np.sqrt(2) / 2 - (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.quad4 = np.array([np.sqrt(2) / 2 - (np.sqrt(2) / 2) * 1j], dtype=np.complex64)
        self.num_good_signals = 0
        self.min_index = 0

        self.window_size = 480
        self.window_slide_dist = 1

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        data_length = range(len(in0))
        # Initialize Per Function Call

        end = 0

        num_windows = int(np.ceil((len(in0) - self.window_size)/self.window_slide_dist) + 1)
        output = in0
        if self.num_good_signals == 0:
            for i in range(num_windows):
                start = i * self.window_slide_dist
                if end > len(in0):
                    end = len(in0)
                else:
                    end = start + self.window_size
                x_win = in0[start:end]
                x_pow = np.sum(np.abs(x_win) ** 2) / self.window_size

                if x_pow > 0.0015:
                    # print('Signal!')

                    test_buffer = np.zeros([1])
                    sigma = np.zeros([1])

                    input_info = in0

                    print("Calibrating ...")
                    for index in self.freq_buffer:
                        # print("Loop: ", index)
                        # print("ListFreq:", self.freq_buffer)
                        # print("Input Data: ", input_info)
                        observed_data = input_info * np.exp(-1j * 2 * np.pi * index * self.sample_period)
                        # print("Observations: ", observed_data)

                        for test in observed_data:
                            if test != 0j:
                                # print("Testing: ", test)
                                temp = min(np.absolute(self.quad1 - test), np.absolute(self.quad2 - test), np.absolute(self.quad3 - test), np.absolute(self.quad4 - test))
                                test_buffer = np.append(test_buffer, temp)
                                # print(test)
                                # print("Minimum Distance: ", min(
                                #     self.quad1 - test, self.quad2 - test, self.quad3 - test, self.quad4 - test))
                                # print("Test Buffer: ", test_buffer)
                        # print("Test Buffer: ", test_buffer)
                        test_buffer = np.delete(test_buffer, 0)
                        mu = np.average(np.absolute(test_buffer))
                        # print("Mu: ", np.average(np.absolute(test_buffer)))
                        # print("Average: ", mu)
                        sigma = np.append(sigma, np.average((np.absolute(test_buffer) - mu) ** 2))

                    sigma = np.delete(sigma, 0)
                    # print("Mu: ", mu)
                    # print("Sigma: ", sigma)
                    self.min_index = np.argmin(sigma)
                    # print("Minimum Index: ", self.min_index)
                    # print("Variance: ", self.min_index)
                    self.num_good_signals = 1
                    print("Estimated Frequency Offset: ", self.freq_buffer[int(self.min_index)], " KHz")
        if self.num_good_signals == 1:
            output = in0 * np.exp(
                -1j * 2 * np.pi * self.freq_buffer[int(self.min_index)] * self.sample_period)
            # print("Estimated Frequency Offset: ", self.freq_buffer[int(self.min_index)], " KHz")
        else:
            output = in0
        out[:] = output

        return len(output_items[0])
