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
from scipy.stats.mstats import gmean
from gnuradio import gr
import random


class SynchronizeAndEstimate(gr.sync_block):

    def __init__(self, case, num_bins, diagnostics, freq_offset, bin_selection, buffer_on, buffer_size, seed_value):
        self.case = 0
        self.case = case
        self.num_bins = float(num_bins)
        self.diagnostics = diagnostics
        self.freq_offset = freq_offset
        self.bin_selection = bin_selection
        self.buffer_size = buffer_size
        self.buffer_on = buffer_on
        self.seed_value = seed_value
        sdr_profile = {0: {'system_scenario': '4G5GSISO-TU',
                           'diagnostic': 1,
                           'wireless_channel': 'Fading',
                           'channel_band': 0.97*960e3,
                           'bin_spacing': 15e3,
                           'channel_profile': 'LTE-TU',
                           'CP_type': 'Normal',
                           'num_ant_txrx': 1,
                           'param_est': 'Estimated',
                           'MIMO_method': 'SpMult',
                           'SNR': 5,
                           'ebno_db': [24],
                           'num_symbols': [48],
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
        self.system_scenario = sdr_profile[self.case]['system_scenario']
        self.diagnostic = sdr_profile[self.case]['diagnostic']
        self.wireless_channel = sdr_profile[self.case]['wireless_channel']
        self.channel_band = sdr_profile[self.case]['channel_band']
        self.bin_spacing = sdr_profile[self.case]['bin_spacing']
        self.channel_profile = sdr_profile[self.case]['channel_profile']
        self.CP_type = sdr_profile[self.case]['CP_type']
        self.num_ant_txrx = sdr_profile[self.case]['num_ant_txrx']
        self.param_est = sdr_profile[self.case]['param_est']
        self.MIMO_method = sdr_profile[self.case]['MIMO_method']  # Make this 0 (or something) for single antenna
        self.SNR = sdr_profile[self.case]['SNR']
        self.ebno_db = sdr_profile[self.case]['ebno_db']
        self.num_symbols = sdr_profile[self.case]['num_symbols']
        self.stream_size = sdr_profile[self.case]['stream_size']

        self.sig_datatype = 'Complex'
        self.phy_chan = 'Data'
        self.modulation_type = 'QPSK'
        self.bits_per_bin = 2
        self.synch_data_pattern = np.array([1, 3])
        self.SNR_type = 'Digital'  # Digital, Analog
        self.ref_sigs = 0.0

        self.NFFT = int(2**(np.ceil(np.log2(round(self.channel_band / self.bin_spacing)))))
        self.fs = self.bin_spacing * self.NFFT
        self.len_CP = int(round(self.NFFT / 4))

        self.num_bins0 = np.floor(self.channel_band / self.bin_spacing)
        num_bins0 = self.num_bins0  # Max number of occupied bins for data
        num_bins1 = 4 * np.floor(num_bins0 / 4)  # Make number of bins a multiple of 4 for MIMO
        if self.diagnostics is True:
            all_bins = np.array(self.bin_selection)
        else:
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

        num_sync_data_patterns = int(np.ceil(self.num_symbols[0] / sum(self.synch_data_pattern)))
        symbol_pattern0 = np.concatenate((np.zeros(self.synch_data_pattern[0]), np.ones(self.synch_data_pattern[1])))
        self.symbol_pattern = np.tile(symbol_pattern0, num_sync_data_patterns)

        self.symbol_length = self.NFFT + self.len_CP

        gr.sync_block.__init__(self,
                               name="SynchronizeAndEstimate",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])

        self.num_sync_bins = self.NFFT - 2
        self.num_of_synchs_and_synch_bins = np.array([self.synch_data_pattern[0], self.num_sync_bins])
        self.total_num_synch_bins = np.product(self.num_of_synchs_and_synch_bins)
        self.prime = 23

        synch_bin_index_from_0 = np.array(range(0, int(self.total_num_synch_bins)))
        synch_bin_index_from_1 = np.array(range(1, int(self.total_num_synch_bins) + 1))
        if self.total_num_synch_bins % 2 == 0:
            self.zadoff_chu = np.exp(-1j * (2 * np.pi / self.total_num_synch_bins) * self.prime * (synch_bin_index_from_0 ** 2 / 2))
        else:
            self.zadoff_chu = np.exp(-1j * (2 * np.pi / self.total_num_synch_bins) * self.prime * (synch_bin_index_from_0 * synch_bin_index_from_1) / 2)

        if self.seed_value != 0:
            index_zadoff_chu = list(range(self.zadoff_chu.shape[0]))
            map_index_position = list(zip(index_zadoff_chu, self.zadoff_chu[:]))

            random.seed(self.seed_value)
            random.shuffle(map_index_position)
            index, self.zadoff_chu = zip(*map_index_position)

        self.used_bin_index = list(range(int(-self.num_sync_bins / 2), 0)) + list(
                range(1, int(self.num_sync_bins / 2) + 1))

        self.used_bins = ((self.NFFT + np.array(self.used_bin_index)) % self.NFFT)
        self.used_bins_synch = self.used_bins.astype(int)  # Same as Caz.used_bins.astype(int) #i
        self.synch_reference = self.zadoff_chu  # i (import file)
        # window: CP to end of symbol
        self.ptr_o = np.array(range(int(self.len_CP), int(self.len_CP + self.NFFT))).astype(int)
        self.ptr_i = self.ptr_o - np.ceil(int(self.len_CP / 2)).astype(int)

        lmax_s = 20
        lmax_d = int(sum(self.symbol_pattern))

        # self.time_synch_ref = np.zeros((self.num_ant_txrx, lmax_s, 2))  # ONE OF THESE 2 WILL BE REMOVED

        self.est_chan_freq_p = np.zeros((self.num_ant_txrx, lmax_s, int(self.NFFT)), dtype=complex)
        self.est_chan_freq_n = np.zeros((self.num_ant_txrx, lmax_s, len(self.used_bins_synch)), dtype=complex)
        self.est_chan_time = np.zeros((self.num_ant_txrx, lmax_s, 3), dtype=complex)
        self.est_synch_freq = np.zeros((self.num_ant_txrx, lmax_s, len(self.used_bins_synch)), dtype=complex)

        if self.num_ant_txrx == 1:
            self.est_data_freq = np.zeros((self.num_ant_txrx, 1, len(self.used_bins_data)), dtype=complex)
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'STCode':
            pass
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'SPMult':
            pass

        # Max length of channel impulse is CP
        self.est_chan_impulse = np.zeros((self.num_ant_txrx, lmax_s, int(self.NFFT)), dtype=complex)

        self.num_of_synchs_and_synch_bins = self.num_of_synchs_and_synch_bins.astype(int)
        self.synch_state = 0

        self.case = case

        self.stride_val = None
        self.correlation_observations = None
        self.start_sample = None
        self.del_mat = None
        self.time_synch_ref = np.zeros((self.num_ant_txrx, 250, 3))  # There are two more in the init.

        self.time_series_data_window = np.zeros(self.NFFT, dtype=complex)
        self.rx_buffer_time_data = None

        self.samp_freq = self.NFFT * self.bin_spacing
        self.samp_period = 1/self.samp_freq

        # Buffer Pointers
        self.start_ptr = 0
        self.end_ptr = buffer_size - 1
        self.current_ptr = 0
        self.current_end_ptr = 0
        self.data_buffer = np.zeros((1, buffer_size)) + 1j * np.zeros((1, buffer_size))
        self.inout = np.zeros((1, buffer_size)) + 1j * np.zeros((1, buffer_size))

        self.dmax_ind_buffer = np.array([0])

    def work(self, input_items, output_items):

        in0 = input_items[0]  # input buffer
        out = output_items[0]  # output buffer

        # Start from the middle of the CP
        if self.num_ant_txrx == 1:
            self.est_data_freq = np.zeros((self.num_ant_txrx, 1, len(self.used_bins_data)), dtype=complex)
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'STCode':
            pass
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'SPMult':
            pass

        input_time_series_data = in0
        input_with_frequency_offset = input_time_series_data

        for index in range(input_time_series_data.shape[0]):
            input_with_frequency_offset[index] = input_time_series_data[index] * np.exp(
                1j * 2 * np.pi * self.freq_offset * self.samp_period * index)

        # num_loops = (len(input_data) - self.window_len) / self.stride_val + 1  # number of windows across rx data
        self.time_synch_ref = np.zeros((self.num_ant_txrx, 250, 3))
        self.stride_val = np.ceil(self.len_CP / 2)

        ptr_frame = 0
        b = 0
        xp = []

        for m in range(1):
            self.correlation_observations = -1

            self.start_sample = (self.len_CP - 4) - 1

            total_loops = int(np.ceil(input_with_frequency_offset.shape[0] / self.stride_val))
            max_correlation_value_buffer = np.zeros(total_loops)

            ptr_adj, loop_count, symbol_count = 0, 0, 0

            tap_delay = 3
            x = np.zeros(tap_delay)
            ptr_synch0 = np.zeros(1000)

            while loop_count <= total_loops:

                if self.correlation_observations == -1:
                    ptr_frame = loop_count * self.stride_val + self.start_sample + ptr_adj
                elif self.correlation_observations < 5:
                    ptr_frame += sum(self.synch_data_pattern) * (int(self.NFFT) + self.len_CP)
                else:
                    ptr_frame = (np.ceil(np.dot(xp[-1:], b) - self.len_CP / 4))[0]

                if (self.num_of_synchs_and_synch_bins[0] - 1) * self.symbol_length + int(self.NFFT) + ptr_frame < input_with_frequency_offset.shape[0]:
                    for i in range(self.num_of_synchs_and_synch_bins[0]):
                        start = int(i * self.symbol_length + ptr_frame)
                        fin = int(i * self.symbol_length + ptr_frame + int(self.NFFT))
                        self.time_series_data_window[i * int(self.NFFT): (i + 1) * int(self.NFFT)] = input_with_frequency_offset[
                                                                                            start:fin]

                    # Take FFT of the window
                    fft_vec = np.zeros((self.num_of_synchs_and_synch_bins[0], int(self.NFFT)), dtype=complex)
                    for i in range(self.num_of_synchs_and_synch_bins[0]):
                        start = i * int(self.NFFT)
                        fin = (i + 1) * int(self.NFFT)
                        fft_vec[i, 0:int(self.NFFT)] = np.fft.fft(self.time_series_data_window[start: fin], int(self.NFFT))

                    synch_symbol_freq_data = fft_vec[:, self.used_bins_synch]
                    synch_symbol_freq_data_vector = np.reshape(synch_symbol_freq_data, (1, synch_symbol_freq_data.shape[0] * synch_symbol_freq_data.shape[1]))

                    pow_est = sum(sum(synch_symbol_freq_data_vector * np.conj(synch_symbol_freq_data_vector))) / synch_symbol_freq_data_vector.shape[1]  # Altered
                    synch_data_normalized = synch_symbol_freq_data_vector / (np.sqrt(pow_est) + 1e-10)

                    bins = self.used_bins_synch[:, None]
                    cp_dels = np.array(range(int(self.len_CP + 1)))[:, None]
                    p_mat0 = np.exp(1j * 2 * (np.pi / self.NFFT) * np.dot(bins, cp_dels.T))

                    p_mat = np.tile(p_mat0, (self.num_of_synchs_and_synch_bins[0], 1))

                    self.del_mat = np.dot(np.conj(self.synch_reference)[None, :], np.dot(np.diag(synch_data_normalized[0]), p_mat))
                    dd = abs(self.del_mat[0, :])
                    max_correlation_value, max_correlation_index = dd.max(0), dd.argmax(0)

                    max_correlation_value_buffer[loop_count] = max_correlation_value

                    if max_correlation_value > 0.5 * synch_data_normalized.shape[1] or self.correlation_observations > -1:
                        if max_correlation_index > np.ceil(0.75 * self.len_CP):
                            if self.correlation_observations == -1:  # 0
                                ptr_adj += np.ceil(0.5 * self.len_CP)
                                ptr_frame = loop_count * self.stride_val + self.start_sample + ptr_adj
                            elif self.correlation_observations < 5:
                                ptr_frame += np.ceil(0.5 * self.len_CP)

                            # Take FFT of the window
                            fft_vec = np.zeros((self.num_of_synchs_and_synch_bins[0], int(self.NFFT)), dtype=complex)
                            for i in range(self.num_of_synchs_and_synch_bins[0]):
                                start = i * int(self.NFFT)
                                fin = (i + 1) * int(self.NFFT)
                                fft_vec[i, 0:int(self.NFFT)] = np.fft.fft(
                                    self.time_series_data_window[start: fin], int(self.NFFT))

                            synch_symbol_freq_data = fft_vec[:, self.used_bins_synch]
                            synch_symbol_freq_data_vector = np.reshape(synch_symbol_freq_data, (1, synch_symbol_freq_data.shape[0] * synch_symbol_freq_data.shape[1]))

                            pow_est = sum(sum(synch_symbol_freq_data_vector * np.conj(synch_symbol_freq_data_vector))) / synch_symbol_freq_data_vector.shape[1]
                            synch_data_normalized = synch_symbol_freq_data_vector / (np.sqrt(pow_est) + 1e-10)

                            bins = self.used_bins_synch[:, None]
                            cp_dels = np.array(range(self.len_CP + 1))[:, None]
                            p_mat0 = np.exp(1j * 2 * (np.pi / int(self.NFFT)) * np.dot(bins, cp_dels.T))

                            p_mat = np.tile(p_mat0, (self.num_of_synchs_and_synch_bins[0], 1))

                            # maybe replace index 0 with m
                            self.del_mat = np.dot(np.conj(self.synch_reference)[None, :],
                                                  np.dot(np.diag(synch_data_normalized[0]), p_mat))
                            dd = abs(self.del_mat[0, :])
                            max_correlation_value, max_correlation_index = dd.max(0), dd.argmax(0)

                            max_correlation_value_buffer[loop_count] = max_correlation_value

                        time_synch_ind = self.time_synch_ref[m, max(self.correlation_observations, 1), 0]

                        if ptr_frame - time_synch_ind > (2 * self.len_CP + int(self.NFFT)) or self.correlation_observations == -1:
                            self.correlation_observations += 1

                            self.time_synch_ref[m, self.correlation_observations, 0] = ptr_frame
                            self.time_synch_ref[m, self.correlation_observations, 1] = max_correlation_index
                            self.time_synch_ref[m, self.correlation_observations, 2] = max_correlation_value

                            ptr_synch0[symbol_count % tap_delay] = sum(self.time_synch_ref[m, self.correlation_observations, 0:2])
                            x[symbol_count % tap_delay] = symbol_count * sum(self.synch_data_pattern)  # No need for +1 on lhs
                            symbol_count += 1

                            x2 = x[0:min(self.correlation_observations, tap_delay)]
                            x_plus = np.concatenate((x2, np.atleast_1d(symbol_count * sum(self.synch_data_pattern))))
                            xp = np.zeros((len(x_plus), 2))
                            xp[:, 0] = np.ones(len(x_plus))
                            xp[:, 1] = x_plus

                            if self.correlation_observations > 3:
                                y = ptr_synch0[0:min(tap_delay, self.correlation_observations)]
                                xl = np.zeros((len(x2), 2))
                                xl[:, 0] = np.ones(len(x2))
                                xl[:, 1] = x2

                                b = np.linalg.lstsq(xl, y)[0]

                            if self.correlation_observations == 0:
                                self.dmax_ind_buffer = np.append(self.dmax_ind_buffer, max_correlation_index)
                                self.dmax_ind_buffer = np.delete(self.dmax_ind_buffer, 0, 0)

                            else:
                                self.dmax_ind_buffer = np.append(self.dmax_ind_buffer, max_correlation_index)

                            if self.dmax_ind_buffer.shape[0] > 3:
                                self.dmax_ind_buffer = self.dmax_ind_buffer[-3:]

                            dmax_ind_processing = 0
                            if dmax_ind_processing == 1:
                                if self.dmax_ind_buffer.shape[0] >= 3:
                                    current_avg_buffer = self.dmax_ind_buffer
                                    average_delay = gmean(current_avg_buffer)
                                    average_delay = np.round(average_delay)

                                    best_index = np.argmin(average_delay)

                                    data_recov0 = np.dot(np.diag(synch_data_normalized[0]), p_mat[:, int(current_avg_buffer[best_index])])  # -1
                                else:
                                    data_recov0 = np.dot(np.diag(synch_data_normalized[0]), p_mat[:, max_correlation_index])
                            else:
                                data_recov0 = np.dot(np.diag(synch_data_normalized[0]), p_mat[:, max_correlation_index])
                            # recovered data with delay removed - DataRecov in MATLAB code

                            h_est1 = np.zeros((int(self.NFFT), 1), dtype=complex)
                            # TmpV1 in MATLAB code
                            # self.SNR += 1e-10
                            data_recov = (data_recov0 * np.conj(self.synch_reference)) / (1 + (1 / self.SNR))

                            h_est00 = np.reshape(data_recov, (data_recov.shape[0], self.num_of_synchs_and_synch_bins[0]))
                            h_est0 = h_est00.T

                            h_est = np.sum(h_est0, axis=0) / (self.num_of_synchs_and_synch_bins[0] + 1e-10)

                            h_est1[self.used_bins_synch, 0] = h_est

                            self.est_chan_freq_p[m, self.correlation_observations, 0:len(h_est1)] = h_est1[:, 0]
                            self.est_chan_freq_n[m, self.correlation_observations, 0:len(h_est)] = h_est

                            h_est_time = np.fft.ifft(h_est1[:, 0], int(self.NFFT))
                            self.est_chan_impulse[m, self.correlation_observations, 0:len(h_est_time)] = h_est_time

                            h_est_ext = np.tile(h_est, (1, self.num_of_synchs_and_synch_bins[0])).T

                            synch_equalized = (data_recov0 * np.conj(h_est_ext[:, 0])) / (
                                    (np.conj(h_est_ext[:, 0]) * h_est_ext[:, 0]) + (1 / (self.SNR + 1e-10)) + 1e-10)
                            self.est_synch_freq[m, self.correlation_observations,
                            0:len(self.used_bins_synch) * self.num_of_synchs_and_synch_bins[0]] = synch_equalized

                loop_count += 1

        if self.num_ant_txrx == 1:
            m = 0  # Just an antenna index
            for p in range(self.correlation_observations):
                for data_sym in range(self.synch_data_pattern[1]):
                    if sum(self.time_synch_ref[m, p, :]) + self.NFFT < input_with_frequency_offset.shape[0]:
                        data_ptr = int(self.time_synch_ref[m, p, 0] + (data_sym + 1) * self.symbol_length)
                        self.rx_buffer_time_data = input_with_frequency_offset[data_ptr: data_ptr + self.NFFT]  # -1

                        fft_vec = np.fft.fft(self.rx_buffer_time_data, self.NFFT)
                        freq_dat0 = fft_vec[self.used_bins_data]
                        p_est = sum(freq_dat0 * np.conj(freq_dat0)) / len(freq_dat0)
                        data_recov0 = freq_dat0 / np.sqrt(p_est)

                        h_est = self.est_chan_freq_p[m, p, self.used_bins_data]

                        del_rotate = np.exp(
                            1j * 2 * (np.pi / self.NFFT) * self.used_bins_data * self.time_synch_ref[m, p, 1])
                        data_recov = np.dot(np.diag(data_recov0), del_rotate)

                        data_equalized = (data_recov * np.conj(h_est)) / (
                                (np.conj(h_est) * h_est) + (1 / self.SNR))

                        if p * self.synch_data_pattern[1] + data_sym == 0:
                            self.est_data_freq[m, p, :] = self.est_data_freq[m, p, :] + data_equalized
                        else:
                            self.est_data_freq = np.vstack((self.est_data_freq[m, :], data_equalized))
                            self.est_data_freq = self.est_data_freq[np.newaxis, :, :]

                        data = self.est_data_freq[m, p, 0:len(self.used_bins_data)]
                        p_est1 = sum(data * np.conj(data)) / (len(data) + 1e-10)

                        self.est_data_freq[
                        m, p * self.synch_data_pattern[1] + data_sym, 0:len(self.used_bins_data)] /= np.sqrt(p_est1)

                        data_out = self.est_data_freq[m, p * self.synch_data_pattern[1] + data_sym,
                                   0:len(self.used_bins_data)]
                        out[0:len(data_out)] = data_out
        return len(output_items[0])
