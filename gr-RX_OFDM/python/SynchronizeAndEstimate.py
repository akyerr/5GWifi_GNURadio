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
from gnuradio import gr

class SynchronizeAndEstimate(gr.sync_block):
    """
    docstring for block SynchronizeAndEstimate
    """
    # def __init__(self, num_ant_txrx, MIMO_method, NFFT, len_CP, SNR, fs, case):
    def __init__(self, case):
        self.case = 0
        self.case = case
        SDR_profile = {0: {'system_scenario': '4G5GSISO-TU',
                            'diagnostic': 1,
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
                            'diagnostic': 0,
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

        self.NFFT = int(2**(np.ceil(np.log2(round(self.channel_band / self.bin_spacing)))))
        self.fs = self.bin_spacing * self.NFFT
        self.len_CP = int(round(self.NFFT / 4))

        self.num_bins0 = np.floor(self.channel_band / self.bin_spacing)

        num_bins0 = self.num_bins0  # Max umber of occupied bins for data
        num_bins1 = 4 * np.floor(num_bins0 / 4)  # Make number of bins a multiple of 4 for MIMO

        # positive and negative bin indices
        all_bins = np.array(list(range(-int(num_bins1 / 2), 0)) + list(range(1, int(num_bins1 / 2) + 1)))
        # positive and negative bin indices
        ref_bins0 = np.random.randint(1, int(num_bins1 / 2) + 1, size=int(np.floor(num_bins1 * self.ref_sigs / 2)))
        ref_bins = np.unique(ref_bins0)
        # positive and negative bin indices
        ref_only_bins = np.sort(np.concatenate((-ref_bins, ref_bins)))  # Bins occupied by pilot (reference) signals
        # positive and negative bin indices - converted to & replaced by positive only in MultiAntennaSystem class
        data_only_bins = np.setdiff1d(all_bins, ref_only_bins)  # Actual bins occupied by data
        self.num_data_bins = len(data_only_bins)
        self.used_bins_data = ((self.NFFT + all_bins) % self.NFFT).astype(int)

        num_synchdata_patterns = int(np.ceil(self.num_symbols[0] / sum(self.synch_data)))
        symbol_pattern0 = np.concatenate((np.zeros(self.synch_data[0]), np.ones(self.synch_data[1])))
        self.symbol_pattern = np.tile(symbol_pattern0, num_synchdata_patterns)

        self.rx_buff_len = self.NFFT + self.len_CP

        # Undefined Use
        # self.UG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))
        # self.SG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))
        # self.VG = np.zeros((self.num_ant_txrx, self.num_ant_txrx, len(self.used_bins_data)))

        gr.sync_block.__init__(self,
                               name="SynchronizeAndEstimate",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])
        self.num_synch_bins = self.NFFT - 2
        self.M = np.array([self.synch_data[0], self.num_synch_bins])
        self.MM = np.product(self.M)
        self.prime = 23

        x0 = np.array(range(0, int(self.MM)))
        x1 = np.array(range(1, int(self.MM) + 1))
        if self.MM % 2 == 0:
            self.ZChu0 = np.exp(-1j * (2 * np.pi / self.MM) * self.prime * (x0 ** 2 / 2))
        else:
            self.ZChu0 = np.exp(-1j * (2 * np.pi / self.MM) * self.prime * (x0 * x1) / 2)

        # print(self.num_used_bins)

        self.used_bins0 = list(range(int(-self.num_synch_bins / 2), 0)) + list(range(1, int(self.num_synch_bins / 2) + 1))
        self.used_bins = ((self.NFFT + np.array(self.used_bins0)) % self.NFFT)
        self.used_bins_synch = self.used_bins.astype(int)  # Same as Caz.used_bins.astype(int) #i
        self.synch_ref = self.ZChu0  # i (import file)
        # window: CP to end of symbol
        self.ptr_o = np.array(range(int(self.len_CP), int(self.len_CP + self.NFFT))).astype(int)
        self.ptr_i = self.ptr_o - np.ceil(int(self.len_CP / 2)).astype(int)

        # lmax_s = int(len(self.symbol_pattern) - sum(self.symbol_pattern))
        # lmax_d = int(sum(self.symbol_pattern))

        lmax_s = 20
        lmax_d = int(sum(self.symbol_pattern))

        # self.time_synch_ref = np.zeros((self.num_ant_txrx, lmax_s, 2))  # ONE OF THESE 2 WILL BE REMOVED

        '''obj.EstChanFreqP=zeros(obj.MIMOAnt,LMAXS,obj.Nfft);
           obj.EstChanFreqN=zeros(obj.MIMOAnt, LMAXS,length(obj.SynchBinsUsed));
           obj.EstChanTim=zeros(obj.MIMOAnt, LMAXS,2);
           obj.EstSynchFreq=zeros(obj.MIMOAnt, LMAXS,length(obj.SynchBinsUsed));'''

        self.est_chan_freq_p = np.zeros((self.num_ant_txrx, lmax_s, int(self.NFFT)), dtype=complex)
        self.est_chan_freq_n = np.zeros((self.num_ant_txrx, lmax_s, len(self.used_bins_synch)), dtype=complex)
        self.est_chan_time = np.zeros((self.num_ant_txrx, lmax_s, 2), dtype=complex)
        self.est_synch_freq = np.zeros((self.num_ant_txrx, lmax_s, len(self.used_bins_synch)), dtype=complex)

        if self.num_ant_txrx == 1:
            self.est_data_freq = np.zeros((self.num_ant_txrx, lmax_d, len(self.used_bins_data)), dtype=complex)
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'STCode':
            pass
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'SPMult':
            pass

        # Max length of channel impulse is CP
        self.est_chan_impulse = np.zeros((self.num_ant_txrx, lmax_s, int(self.NFFT)), dtype=complex)

        self.M = self.M.astype(int)
        self.synch_data = self.synch_data
        self.synch_state = 0

        self.case = case

        self.stride_val = None
        self.corr_obs = None
        self.start_samp = None
        self.del_mat = None
        self.time_synch_ref = np.zeros((self.num_ant_txrx, 250, 3))  # There are two more in the init.

        self.rx_buffer_time = np.zeros(self.NFFT, dtype=complex)
        self.rx_buffer_time_data = None
    # def write_to_file(self, file_name, var_to_write, var_string):
    #     f = open(file_name, 'a')
    #     f.write('\n' + var_string + str(var_to_write))
    #     f.close()

    def work(self, input_items, output_items):
        in0 = input_items[0]  # input buffer
        out = output_items[0]  # output buffer

        # print(self.used_bins_data)
        # print(self.num_data_bins)

        # print(in0.shape)
        # print(in0[0:10])

        # self.write_to_file(file_name, self.count, 'Work call: ')

        # input_data = in0  # read input data from buffer

        # Start from the middle of the CP
        rx_buffer_time0 = in0
        # print(rx_buffer_time0.shape)
        # rx_buffer_time = rx_buffer_time0[:, ]

        # num_loops = (len(input_data) - self.window_len) / self.stride_val + 1  # number of windows across rx data

        self.stride_val = np.ceil(self.len_CP / 2)

        ptr_frame = 0
        b = 0
        xp = []

        for m in range(1):
            self.corr_obs = -1

            # chan_q = self.genie_chan_time[m, 0, :]  # 2048
            self.start_samp = (self.len_CP - 4) - 1

            total_loops = int(np.ceil(rx_buffer_time0.shape[0] / self.stride_val))
            # print(total_loops)
            d_long = np.zeros(total_loops)

            ptr_adj, loop_count, sym_count = 0, 0, 0

            tap_delay = 5
            x = np.zeros(tap_delay)
            ptr_synch0 = np.zeros(1000)
            # data_recov = [0 + 0j]
            while loop_count <= total_loops:
                # print(loop_count)
                if self.corr_obs == -1:
                    ptr_frame = loop_count * self.stride_val + self.start_samp + ptr_adj
                elif self.corr_obs < 5:
                    ptr_frame += sum(self.synch_data) * (int(self.NFFT) + self.len_CP)
                else:
                    ptr_frame = (np.ceil(np.dot(xp[-1:], b) - self.len_CP / 4))[0]

                # print(rx_buffer_time0.shape[1])
                if (self.M[0] - 1) * self.rx_buff_len + int(self.NFFT) + ptr_frame < rx_buffer_time0.shape[0]:
                    # print('I am here')
                    # if (self.MM[0] - 1)*self.rx_buff_len + self.NFFT + ptr_frame - 1 < rx_buffer_time0.shape[1]:
                    for i in range(self.M[0]):
                        # print(i)
                        start = int(i * self.rx_buff_len + ptr_frame)
                        fin = int(i * self.rx_buff_len + ptr_frame + int(self.NFFT))
                        self.rx_buffer_time[i * int(self.NFFT): (i + 1) * int(self.NFFT)] = rx_buffer_time0[start:fin]

                    # Take FFT of the window
                    fft_vec = np.zeros((self.M[0], int(self.NFFT)), dtype=complex)
                    for i in range(self.M[0]):
                        start = i * int(self.NFFT)
                        fin = (i + 1) * int(self.NFFT)
                        fft_vec[i, 0:int(self.NFFT)] = np.fft.fft(self.rx_buffer_time[start: fin], int(self.NFFT))

                    # print('Synch bins', self.used_bins_synch)
                    # print('No of synch bins', self.used_bins_synch.shape)
                    synch_dat00 = fft_vec[:, self.used_bins_synch]
                    synch_dat0 = np.reshape(synch_dat00, (1, synch_dat00.shape[0] * synch_dat00.shape[1]))
                    pow_est = sum(sum(synch_dat0 * np.conj(synch_dat0))).real / synch_dat0.shape[1]
                    synch_dat = synch_dat0 / np.sqrt(pow_est)

                    # from transmit antenna 1 only?
                    # chan_freq0 = np.reshape(self.channel_freq[m, 0, self.used_bins_synch],
                    #                         (1, np.size(self.used_bins_synch)))
                    #
                    # chan_freq = np.tile(chan_freq0, (1, self.M[0]))
                    # print('I am here')
                    bins = self.used_bins_synch[:, None]
                    cp_dels = np.array(range(int(self.len_CP + 1)))[:, None]
                    p_mat0 = np.exp(1j * 2 * (np.pi / self.NFFT) * np.dot(bins, cp_dels.T))

                    p_mat = np.tile(p_mat0, (self.M[0], 1))

                    # maybe replace index 0 with m
                    self.del_mat = np.dot(np.conj(self.synch_ref)[None, :], np.dot(np.diag(synch_dat[0]), p_mat))
                    dd = abs(self.del_mat[0, :])
                    dmax, dmax_ind0 = dd.max(0), dd.argmax(0)
                    dmax_ind = dmax_ind0 - 1
                    d_long[loop_count] = dmax
                    # print('no')
                    # print(dmax)
                    if dmax > 0.5 * synch_dat.shape[1] or self.corr_obs > -1:
                        # print('I am here')
                        if dmax_ind > np.ceil(0.75 * self.len_CP):
                            if self.corr_obs == -1: # 0
                                ptr_adj += np.ceil(0.5 * self.len_CP)
                                ptr_frame = loop_count * self.stride_val + self.start_samp + ptr_adj
                            elif self.corr_obs < 5:
                                ptr_frame += np.ceil(0.5 * self.len_CP)

                            # Take FFT of the window
                            fft_vec = np.zeros((self.M[0], int(self.NFFT)), dtype=complex)
                            for i in range(self.M[0]):
                                start = i * int(self.NFFT)
                                fin = (i + 1) * int(self.NFFT)
                                fft_vec[i, 0:int(self.NFFT)] = np.fft.fft(self.rx_buffer_time[start: fin], int(self.NFFT))

                            synch_dat00 = fft_vec[:, self.used_bins_synch]
                            synch_dat0 = np.reshape(synch_dat00, (1, synch_dat00.shape[0] * synch_dat00.shape[1]))
                            pow_est = sum(sum(synch_dat0 * np.conj(synch_dat0))).real / synch_dat0.shape[1]
                            synch_dat = synch_dat0 / np.sqrt(pow_est)

                            # from transmit antenna 1 only?
                            # chan_freq0 = np.reshape(self.channel_freq[m, 0, self.used_bins_synch],
                            #                         (1, np.size(self.used_bins_synch)))

                            # chan_freq = np.tile(chan_freq0, (1, self.M[0]))

                            bins = self.used_bins_synch[:, None]
                            cp_dels = np.array(range(self.len_CP + 1))[:, None]
                            p_mat0 = np.exp(1j * 2 * (np.pi / int(self.NFFT)) * np.dot(bins, cp_dels.T))

                            p_mat = np.tile(p_mat0, (self.M[0], 1))

                            # maybe replace index 0 with m
                            self.del_mat = np.dot(np.conj(self.synch_ref)[None, :],
                                                  np.dot(np.diag(synch_dat[0]), p_mat))
                            dd = abs(self.del_mat[0, :])
                            dmax, dmax_ind0 = dd.max(0), dd.argmax(0)
                            dmax_ind = dmax_ind0 - 1
                            d_long[loop_count] = dmax

                        time_synch_ind = self.time_synch_ref[m, max(self.corr_obs, 1), 0]

                        if ptr_frame - time_synch_ind > (2 * self.len_CP + int(self.NFFT)) or self.corr_obs == -1:
                            self.corr_obs += 1

                            self.time_synch_ref[m, self.corr_obs, 0] = ptr_frame
                            self.time_synch_ref[m, self.corr_obs, 1] = dmax_ind
                            self.time_synch_ref[m, self.corr_obs, 2] = dmax

                            ptr_synch0[sym_count % tap_delay] = sum(self.time_synch_ref[m, self.corr_obs, 0:2])
                            x[sym_count % tap_delay] = sym_count * sum(self.synch_data)  # No need for +1 on lhs
                            sym_count += 1

                            x2 = x[0:min(self.corr_obs, tap_delay)]
                            x_plus = np.concatenate((x2, np.atleast_1d(sym_count * sum(self.synch_data))))
                            xp = np.zeros((len(x_plus), 2))
                            xp[:, 0] = np.ones(len(x_plus))
                            xp[:, 1] = x_plus

                            if self.corr_obs > 3:
                                y = ptr_synch0[0:min(tap_delay, self.corr_obs)]
                                # print(y)
                                X = np.zeros((len(x2), 2))
                                X[:, 0] = np.ones(len(x2))
                                X[:, 1] = x2

                                b = np.linalg.lstsq(X, y)[0]
                                # print(b)

                            # recovered data with delay removed - DataRecov in MATLAB code
                            data_recov0 = np.dot(np.diag(synch_dat[0]), p_mat[:, dmax_ind])  #

                            h_est1 = np.zeros((int(self.NFFT), 1), dtype=complex)
                            # TmpV1 in MATLAB code
                            data_recov = (data_recov0 * np.conj(self.synch_ref)) / (1 + (1 / self.SNR))

                            h_est00 = np.reshape(data_recov, (data_recov.shape[0], self.M[0]))
                            h_est0 = h_est00.T

                            h_est = np.sum(h_est0, axis=0) / (self.M[0])

                            h_est1[self.used_bins_synch, 0] = h_est

                            self.est_chan_freq_p[m, self.corr_obs, 0:len(h_est1)] = h_est1[:, 0]
                            self.est_chan_freq_n[m, self.corr_obs, 0:len(h_est)] = h_est

                            # if sys_model.diagnostic == 1 and loop_count == 0:
                            #     xax = np.array(range(0, self.NFFT)) * sys_model.fs/self.NFFT
                            #     yax1 = 20*np.log10(abs(h_est1))
                            #     yax2 = 20*np.log10(abs(np.fft.fft(chan_q, self.NFFT)))
                            #
                            #     plt.plot(xax, yax1, 'r')
                            #     plt.plot(xax, yax2, 'b')
                            #     plt.show()

                            h_est_time = np.fft.ifft(h_est1[:, 0], int(self.NFFT))
                            self.est_chan_impulse[m, self.corr_obs, 0:len(h_est_time)] = h_est_time

                            h_est_ext = np.tile(h_est, (1, self.M[0])).T
                            # print("equalized synch")

                            synch_equalized = (data_recov0 * np.conj(h_est_ext[:, 0])) / (
                                        (np.conj(h_est_ext[:, 0]) * h_est_ext[:, 0]) + (1 / self.SNR))
                            self.est_synch_freq[m, self.corr_obs,
                                                0:len(self.used_bins_synch) * self.M[0]] = synch_equalized

                            # print(synch_equalized[0:10])
                            # print(synch_equalized.shape)
                            # out[0:len(synch_equalized)] = synch_equalized

                loop_count += 1


        # return len(output_items[0])


        if self.num_ant_txrx == 1:
            m = 0  # Just an antenna index
            for p in range(self.corr_obs + 1):
                for data_sym in range(self.synch_data[1]):
                    if sum(self.time_synch_ref[m, p, :]) + self.NFFT < rx_buffer_time0.shape[0]:
                        data_ptr = int(self.time_synch_ref[m, p, 0] + (data_sym + 1) * self.rx_buff_len)
                        self.rx_buffer_time_data = rx_buffer_time0[data_ptr: data_ptr + self.NFFT]  # -1

                        fft_vec = np.fft.fft(self.rx_buffer_time_data, self.NFFT)

                        freq_dat0 = fft_vec[self.used_bins_data]

                        p_est = sum(freq_dat0 * np.conj(freq_dat0)) / len(freq_dat0)

                        p_est += 1e-10

                        data_recov0 = freq_dat0 / np.sqrt(p_est)

                        # if self.param_est == 'Estimated':
                        #     print('hello')
                        #     h_est = self.est_chan_freq_p[m, p, self.used_bins_data]
                        # elif self.param_est == 'Ideal':
                        #     print('bye') # GENIE
                        #     h_est = self.h_f[m, 0, :]
                        h_est = self.est_chan_freq_p[m, p, self.used_bins_data]

                        del_rotate = np.exp(
                            1j * 2 * (np.pi / self.NFFT) * self.used_bins_data * self.time_synch_ref[m, p, 1])
                        data_recov = np.dot(np.diag(data_recov0), del_rotate)

                        data_equalized = (data_recov * np.conj(h_est)) / (
                                    (np.conj(h_est) * h_est) + (1 / self.SNR))
                        self.est_data_freq[m, p * self.synch_data[1] + data_sym,
                                           0:len(self.used_bins_data)] = data_equalized

                        data = self.est_data_freq[m, p, 0:len(self.used_bins_data)]
                        p_est1 = sum(data * np.conj(data)) / len(data)

                        self.est_data_freq[m, p * self.synch_data[1] + data_sym,0:len(self.used_bins_data)] /= np.sqrt(p_est1)


                        data_out = self.est_data_freq[m, p * self.synch_data[1] + data_sym, 0:len(self.used_bins_data)]
                        # print(data_out.shape)
                        out[0:len(data_out)] = data_out
        # print('Returning {}'.format(len(output_items[0])))
        # print('Start of buffer {}'.format(out[0:10]))
        # print('End of buffer {}'.format(out[-10:]))
        return len(output_items[0])
