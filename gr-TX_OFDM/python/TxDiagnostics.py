#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2019 gr-TX_OFDM author.
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

import numpy
import pickle
from gnuradio import gr


class TxDiagnostics(gr.sync_block):
    """
    docstring for block TxDiagnostics
    """
    def __init__(self, case, pickle_dir):
        gr.sync_block.__init__(self,
                               name="TxDiagnostics",
                               in_sig=None,
                               out_sig=[numpy.complex64])

        with open('{}/tx_data_{}.pckl'.format(pickle_dir, case)) as f:
            self.tx_data = pickle.load(f)

        self.work_call = 0

    def work(self, input_items, output_items):
        out = output_items[0]

        # if out[:].shape == self.tx_data.shape[1]:
        if 50 <= self.work_call <= 600:
            # print("Generating Signal...")
            if out[:].shape[0] < self.tx_data.shape[1]:
                out[:] = numpy.zeros(out[:].shape[0])
            else:
                out[0:self.tx_data.shape[1]] = self.tx_data[0, :]
        else:
            # print("Generating Zeros...")
            out[:] = numpy.zeros(out[:].shape[0])

        self.work_call += 1
        # print("Work Call: ", self.work_call)

        return len(output_items[0])
