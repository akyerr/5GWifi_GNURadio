#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2019 <+YOU OR YOUR COMPANY+>.
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
import csv
import pickle
import datetime
from gnuradio import gr

class BitRecovery(gr.sync_block):
    """
     docstring for block BitRecovery
     """

    def __init__(self, case, directory_name, diagnostics):
        gr.sync_block.__init__(self,
                               name="BitRecovery",
                               in_sig=[np.complex64],
                               out_sig=None)

        self.case = 0
        self.case = case
        SDR_profile = {0: {'system_scenario': '4G5GSISO-TU',
                            'diagnostic': diagnostics,
                            'wireless_channel': 'Fading',
                            'channel_band': 0.97*960e3,
                            'bin_spacing': 15e3,
                            'channel_profile': 'LTE-TU',
                            'CP_type': 'Normal',
                            'num_ant_txrx': 1,
                            'param_est': 'Estimated',
                            'MIMO_method': 'SpMult',
                            'SNR': 100,
                            'ebno_db': [100, 100, 100, 100, 100, 100, 100, 100, 100],
                            'num_symbols': [48, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
                            'stream_size': 1},
                        1: {'system_scenario': 'WIFIMIMOSM-A',
                            'diagnostic': diagnostics,
                            'wireless_channel': 'Fading',
                            'channel_band': 0.9 * 20e6,
                            'bin_spacing': 312.5e3,
                            'channel_profile': 'Indoor A',
                            'CP_type': 'Extended',
                            'num_ant_txrx': 2,
                            'param_est': 'Ideal',
                            'MIMO_method': 'SpMult',
                            'SNR': 50,
                            'ebno_db': [6, 7, 8, 9, 10, 14, 16, 20, 24],
                            'num_symbols': [10, 10, 10, 10, 10, 10, 10, 10, 10],
                            'stream_size': 2}}
        self.system_scenario = SDR_profile[self.case]['system_scenario']
        self.diagnostic = SDR_profile[self.case]['diagnostic']
        self.wireless_channel = SDR_profile[self.case]['wireless_channel']
        self.channel_band = SDR_profile[self.case]['channel_band']
        self.bin_spacing = SDR_profile[self.case]['bin_spacing']
        self.channel_profile = SDR_profile[self.case]['channel_profile']
        self.CP_type = SDR_profile[self.case]['CP_type']
        self.num_ant_txrx = SDR_profile[self.case]['num_ant_txrx']
        self.param_est = SDR_profile[self.case]['param_est']
        self.MIMO_method = SDR_profile[self.case]['MIMO_method']  # Make this 0 (or something) for single antenna
        self.SNR = SDR_profile[self.case]['SNR']
        self.ebno_db = SDR_profile[self.case]['ebno_db']
        self.num_symbols = SDR_profile[self.case]['num_symbols']
        self.stream_size = SDR_profile[self.case]['stream_size']

        self.sig_datatype = 'Complex'
        self.phy_chan = 'Data'
        self.modulation_type = 'QPSK'
        self.bits_per_bin = 2
        self.synch_data = np.array([1, 3])
        self.SNR_type = 'Digital'  # Digital, Analog
        self.ref_sigs = 0.0

        self.pickle_dir = directory_name
        self.diagnostics = diagnostics

        self.symbol_count = 0
        self.mapping_type = 'QPSK'
        self.EC_code = None
        self.data_in = None
        self.softbit0 = None
        self.softbit1 = None
        self.hardbit = None
        self.num_bins0 = np.floor(self.channel_band / self.bin_spacing)
        num_bins0 = self.num_bins0  # Max umber of occupied bins for data
        num_bins1 = 4 * np.floor(num_bins0 / 4)  # Make number of bins a multiple of 4 for MIMO

        self.NFFT = int(2 ** (np.ceil(np.log2(round(self.channel_band / self.bin_spacing)))))
        self.count = 0
        # positive and negative bin indices
        all_bins = np.array(list(range(-int(num_bins1 / 2), 0)) + list(range(1, int(num_bins1 / 2) + 1)))
        self.used_bins_data = ((self.NFFT + all_bins) % self.NFFT).astype(int)

        self.count = 0


    def work(self, input_items, output_items):

        in0 = input_items[0]
        # print("In", in0[100:110])

        # print(in0[60:100])

        data_in = in0[:, np.newaxis]

        num_bits = len(data_in)*2
        num_symbs = len(data_in)

        llrp0 = np.zeros(num_bits, dtype=float)
        llrp1 = np.zeros(num_bits, dtype=float)

        cmplx_phsrs = np.exp(1j * 2 * (np.pi / 8) * np.array([1, -1, 3, 5]))
        cmplx_phsrsext = np.tile(cmplx_phsrs, (num_symbs, 1))
        data_ext = np.tile(data_in, (1, 4))

        zobs = data_ext - cmplx_phsrsext

        dminind = np.argmin(abs(zobs), 1)
        dmin = np.min(abs(zobs), 1)

        dz = cmplx_phsrs[dminind]
        ez = np.zeros(len(data_in), dtype=complex)

        for ii in range(len(data_in)):
            ez[ii] = data_in[ii][0] - dz[ii]

        sigma0 = (1/np.sqrt(2)) * np.mean(abs(dmin))
        d_factor = 1 / sigma0 ** 2

        K = 2 / np.sqrt(2)

        for kk in range(len(data_in)):
            if data_in[kk].real >= 0 and data_in[kk].imag >= 0:
                llrp0[2 * kk] = -0.5 * abs(ez[kk].real)
                llrp1[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                llrp0[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                llrp1[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))

            elif data_in[kk].real <= 0 and data_in[kk].imag >= 0:
                llrp0[2 * kk] = -0.5 * abs(ez[kk].real)
                llrp1[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                llrp1[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                llrp0[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))
            elif data_in[kk].real <= 0 and data_in[kk].imag <= 0:
                llrp1[2 * kk] = -0.5 * abs(ez[kk].real)
                llrp0[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                llrp1[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                llrp0[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))
            elif data_in[kk].real >= 0 and data_in[kk].imag <= 0:
                llrp1[2 * kk] = -0.5 * abs(ez[kk].real)
                llrp0[2 * kk] = -0.5 * (K - abs(ez[kk].real))

                llrp0[2 * kk + 1] = -0.5 * abs(ez[kk].imag)
                llrp1[2 * kk + 1] = -0.5 * (K - abs(ez[kk].imag))

        llrp0 *= d_factor
        llrp1 *= d_factor

        softbit0 = llrp0
        softbit1 = llrp1

        hardbits = np.array(0.5 * (np.sign(softbit1 - softbit0) + 1.)).astype(int)

        print(hardbits.shape)

        # est_bits = hardbit[:, np.newaxis]

        # print(est_bits[])

        if self.diagnostic == 1 and self.count == 5:

            print('Writing bits to file...')
            with open('{}/rx_bits_{}.pckl'.format(self.pickle_dir, self.case), 'wb') as f:
                pickle.dump(hardbits, f, protocol=2)


        self.count += 1
        return len(input_items[0])

