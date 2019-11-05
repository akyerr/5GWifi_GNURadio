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
import sys


class SynchronizeAndEstimate(gr.sync_block):
    """
    docstring for block SynchronizeAndEstimate
    """
    # def __init__(self, num_ant_txrx, MIMO_method, NFFT, len_CP, SNR, fs, case):
    def __init__(self, case, num_bins, diagnostics, freq_offset, bin_selection, buffer_on, buffer_size, seed_value, snr, corr_ind_processing):
        self.case = 0
        self.case = case
        self.num_bins = float(num_bins)
        self.diagnostics = diagnostics
        self.freq_offset = freq_offset
        self.bin_selection = bin_selection
        self.buffer_size = buffer_size
        self.buffer_on = buffer_on
        self.seed_value = seed_value
        self.correlation_ind_processing = corr_ind_processing
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
        self.SNR = snr
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
        ref_bins0 = np.random.randint(1, int(num_bins1 / 2) + 1, size=int(np.floor(num_bins1 * self.ref_sigs / 2)))
        ref_bins = np.unique(ref_bins0)

        # positive and negative bin indices
        ref_only_bins = np.sort(np.concatenate((-ref_bins, ref_bins)))  # Bins occupied by pilot (reference) signals
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
        self.total_num_of_synch_bins = np.product(self.num_of_synchs_and_synch_bins)
        self.prime = 23

        synch_bin_index_from_0 = np.array(range(0, int(self.total_num_of_synch_bins)))
        synch_bin_index_from_1 = np.array(range(1, int(self.total_num_of_synch_bins) + 1))
        if self.total_num_of_synch_bins % 2 == 0:
            self.zadoff_chu = np.exp(-1j * (2 * np.pi / self.total_num_of_synch_bins) * self.prime * (synch_bin_index_from_0 ** 2 / 2))
        else:
            self.zadoff_chu = np.exp(-1j * (2 * np.pi / self.total_num_of_synch_bins) * self.prime * (synch_bin_index_from_0 * synch_bin_index_from_1) / 2)

        if self.seed_value != 0:
            index_zadoff_chu = list(range(self.zadoff_chu.shape[0]))
            map_index_position = list(zip(index_zadoff_chu, self.zadoff_chu[:]))

            random.seed(self.seed_value)
            random.shuffle(map_index_position)
            index, self.zadoff_chu = zip(*map_index_position)

        self.synch_used_bin_index_n_p = list(range(int(-self.num_sync_bins / 2), 0)) + list(
                range(1, int(self.num_sync_bins / 2) + 1))

        self.synch_used_bin_index_all_positive = ((self.NFFT + np.array(self.synch_used_bin_index_n_p)) % self.NFFT)
        self.synch_used_bins = self.synch_used_bin_index_all_positive.astype(int)  # Same as Caz.synch_used_bin_index_positive.astype(int) #i
        self.synch_reference = self.zadoff_chu  # i (import file)
        # window: CP to end of symbol

        estimated_buffer_size = 20
        self.est_chan_freq_p = np.zeros((self.num_ant_txrx, estimated_buffer_size, int(self.NFFT)), dtype=complex)
        self.est_chan_freq_n = np.zeros((self.num_ant_txrx, estimated_buffer_size, len(self.synch_used_bins)), dtype=complex)
        self.est_chan_time = np.zeros((self.num_ant_txrx, estimated_buffer_size, 3), dtype=complex)
        self.est_synch_freq = np.zeros((self.num_ant_txrx, estimated_buffer_size, len(self.synch_used_bins)), dtype=complex)

        if self.num_ant_txrx == 1:
            self.est_data_freq = np.zeros((self.num_ant_txrx, 1, len(self.used_bins_data)), dtype=complex)
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'STCode':
            pass
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'SPMult':
            pass

        # Max length of channel impulse is CP
        self.est_chan_impulse = np.zeros((self.num_ant_txrx, estimated_buffer_size, int(self.NFFT)), dtype=complex)

        self.num_of_synchs_and_synch_bins = self.num_of_synchs_and_synch_bins.astype(int)
        self.synch_state = 0

        self.case = case

        self.stride_val = None
        self.correlation_observations = None
        self.start_sample = None
        self.correlation_matrix = None
        self.correlation_frame_index_value_buffer = np.zeros((self.num_ant_txrx, 250, 3))  # There are two more in the init.

        self.time_series_data_window = np.zeros(self.NFFT, dtype=complex)
        self.rx_buffer_time_data = None

        self.sample_freq = self.NFFT * self.bin_spacing
        self.sample_period = 1 / self.sample_freq

        # Internal Buffer Pointers
        self.start_ptr = 0  # Input Buffer Pointer
        self.end_ptr = buffer_size - 1  # Input Buffer Pointer
        self.current_ptr = 0  # Output Buffer Pointer
        self.current_end_ptr = 0  # Output Buffer Pointer
        self.data_buffer = np.zeros((1, buffer_size)) + 1j * np.zeros((1, buffer_size))  # Initializing Buffer
        self.inout = np.zeros((1, buffer_size)) + 1j * np.zeros((1, buffer_size))  # Initializing Buffer

        self.max_correlation_index_buffer = np.array([0])

    def work(self, input_items, output_items):

        in0 = input_items[0]  # input buffer
        out = output_items[0]  # output buffer

        if self.num_ant_txrx == 1:
            self.est_data_freq = np.zeros((self.num_ant_txrx, 1, len(self.used_bins_data)), dtype=complex)
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'STCode':
            pass
        elif self.num_ant_txrx == 2 and self.MIMO_method == 'SPMult':
            pass

        input_time_series_data = in0
        input_with_freq_offset = input_time_series_data

        for index in range(input_time_series_data.shape[0]):
            input_with_freq_offset[index] = input_time_series_data[index] * np.exp(
                1j * 2 * np.pi * self.freq_offset * self.sample_period * index)

        self.correlation_frame_index_value_buffer = np.zeros((self.num_ant_txrx, 250, 3))
        self.stride_val = np.ceil(self.len_CP / 2)

        ptr_frame = 0
        m_c_coefficients = 0
        x_symbol_count_lookahead = []

        for antenna_index in range(1):
            self.correlation_observations = -1
            self.start_sample = (self.len_CP - 4) - 1
            total_loops = int(np.ceil(input_with_freq_offset.shape[0] / self.stride_val))
            correlations_val_vector = np.zeros(total_loops)

            ptr_adj, loop_count, frame_count = 0, 0, 0

            tap_delay = 5
            symbol_count = np.zeros(tap_delay)
            corrected_ptr = np.zeros(1000)

            while loop_count <= total_loops:
                if self.correlation_observations == -1:
                    ptr_frame = loop_count * self.stride_val + self.start_sample + ptr_adj
                elif self.correlation_observations < 5:
                    ptr_frame += sum(self.synch_data_pattern) * (int(self.NFFT) + self.len_CP)
                else:
                    ptr_frame = (np.ceil(np.dot(x_symbol_count_lookahead[-1:], m_c_coefficients) - self.len_CP / 4))[0]

                if (self.num_of_synchs_and_synch_bins[0] - 1) * self.symbol_length + int(self.NFFT) + ptr_frame < input_with_freq_offset.shape[0]:
                    for i in range(self.num_of_synchs_and_synch_bins[0]):
                        start = int(i * self.symbol_length + ptr_frame)
                        fin = int(i * self.symbol_length + ptr_frame + int(self.NFFT))
                        self.time_series_data_window[i * int(self.NFFT): (i + 1) * int(self.NFFT)] = input_with_freq_offset[
                                                                                            start:fin]
                    # Take FFT of the window
                    fft_vec = np.zeros((self.num_of_synchs_and_synch_bins[0], int(self.NFFT)), dtype=complex)
                    for i in range(self.num_of_synchs_and_synch_bins[0]):
                        start = i * int(self.NFFT)
                        fin = (i + 1) * int(self.NFFT)
                        fft_vec[i, 0:int(self.NFFT)] = np.fft.fft(self.time_series_data_window[start: fin], int(self.NFFT))

                    synch_freq_data = fft_vec[:, self.synch_used_bins]
                    synch_freq_data_vector = np.reshape(synch_freq_data, (1, synch_freq_data.shape[0] * synch_freq_data.shape[1]))
                    synch_pow_est = sum(sum(synch_freq_data_vector * np.conj(synch_freq_data_vector))).real / synch_freq_data_vector.shape[1]
                    synch_freq_data_normalized = synch_freq_data_vector / (np.sqrt(synch_pow_est) + 1e-10)

                    bin_index = self.synch_used_bins[:, None]
                    cp_delays = np.array(range(int(self.len_CP + 1)))[:, None]
                    delay_matrix = np.exp(1j * 2 * (np.pi / self.NFFT) * np.dot(bin_index, cp_delays.T))

                    tiled_delay_matrix = np.tile(delay_matrix, (self.num_of_synchs_and_synch_bins[0], 1))

                    self.correlation_matrix = np.dot(np.conj(self.synch_reference)[None, :], np.dot(np.diag(synch_freq_data_normalized[0]), tiled_delay_matrix))
                    abs_correlation_matrix = abs(self.correlation_matrix[0, :])
                    correlation_value, correlation_index = abs_correlation_matrix.max(0), abs_correlation_matrix.argmax(0)
                    correlations_val_vector[loop_count] = correlation_value

                    if correlation_value > 0.50 * synch_freq_data_normalized.shape[1] or self.correlation_observations > -1:
                        if correlation_index > np.ceil(0.50 * self.len_CP):
                            if self.correlation_observations == -1:  # 0
                                ptr_adj += np.ceil(0.5 * self.len_CP)
                                ptr_frame = loop_count * self.stride_val + self.start_sample + ptr_adj
                            elif self.correlation_observations < 5:
                                ptr_frame += np.ceil(0.5 * self.len_CP)

                            fft_vec = np.zeros((self.num_of_synchs_and_synch_bins[0], int(self.NFFT)), dtype=complex)
                            for i in range(self.num_of_synchs_and_synch_bins[0]):
                                start = i * int(self.NFFT)
                                fin = (i + 1) * int(self.NFFT)
                                fft_vec[i, 0:int(self.NFFT)] = np.fft.fft(
                                    self.time_series_data_window[start: fin], int(self.NFFT))

                            synch_freq_data = fft_vec[:, self.synch_used_bins]
                            synch_freq_data_vector = np.reshape(synch_freq_data, (1, synch_freq_data.shape[0] * synch_freq_data.shape[1]))
                            synch_pow_est = sum(sum(synch_freq_data_vector * np.conj(synch_freq_data_vector))).real / synch_freq_data_vector.shape[1]
                            synch_freq_data_normalized = synch_freq_data_vector / (np.sqrt(synch_pow_est) + 1e-10)

                            bin_index = self.synch_used_bins[:, None]
                            cp_delays = np.array(range(self.len_CP + 1))[:, None]
                            print("Result: ", np.dot(bin_index, cp_delays.T))
                            delay_matrix = np.exp(1j * 2 * (np.pi / int(self.NFFT)) * np.dot(bin_index, cp_delays.T))
                            tiled_delay_matrix = np.tile(delay_matrix, (self.num_of_synchs_and_synch_bins[0], 1))

                            self.correlation_matrix = np.dot(np.conj(self.synch_reference)[None, :], np.dot(np.diag(synch_freq_data_normalized[0]), tiled_delay_matrix))
                            abs_correlation_matrix = abs(self.correlation_matrix[0, :])
                            correlation_value, correlation_index = abs_correlation_matrix.max(0), abs_correlation_matrix.argmax(0)
                            correlations_val_vector[loop_count] = correlation_value

                        time_synch_ind = self.correlation_frame_index_value_buffer[antenna_index, max(self.correlation_observations, 1), 0]

                        if ptr_frame - time_synch_ind > (sum(self.synch_data_pattern) * (self.len_CP + int(self.NFFT))) or self.correlation_observations == -1:
                            self.correlation_observations += 1

                            self.correlation_frame_index_value_buffer[antenna_index, self.correlation_observations, 0] = ptr_frame
                            self.correlation_frame_index_value_buffer[antenna_index, self.correlation_observations, 1] = correlation_index
                            self.correlation_frame_index_value_buffer[antenna_index, self.correlation_observations, 2] = correlation_value

                            corrected_ptr[frame_count % tap_delay] = sum(self.correlation_frame_index_value_buffer[antenna_index, self.correlation_observations, 0:2])
                            symbol_count[frame_count % tap_delay] = frame_count * sum(self.synch_data_pattern)  # No need for +1 on lhs
                            print("X: ", symbol_count)
                            frame_count += 1

                            symbol_count_current = symbol_count[0:min(self.correlation_observations, tap_delay)]
                            print("X2: ", symbol_count_current)
                            symbol_count_lookahead = np.concatenate((symbol_count_current, np.atleast_1d(frame_count * sum(self.synch_data_pattern))))
                            print("Xplus: ", symbol_count_lookahead)
                            x_symbol_count_lookahead = np.zeros((len(symbol_count_lookahead), 2))
                            x_symbol_count_lookahead[:, 0] = np.ones(len(symbol_count_lookahead))
                            x_symbol_count_lookahead[:, 1] = symbol_count_lookahead

                            if self.correlation_observations > 3:
                                y_time_series_current_ptr = corrected_ptr[0:min(tap_delay, self.correlation_observations)]
                                x_symbol_count_current = np.zeros((len(symbol_count_current), 2))
                                x_symbol_count_current[:, 0] = np.ones(len(symbol_count_current))
                                x_symbol_count_current[:, 1] = symbol_count_current
                                print("x_symbol_count_current: ", x_symbol_count_current, " y_time_series_current_ptr: ", y_time_series_current_ptr)
                                m_c_coefficients = np.linalg.lstsq(x_symbol_count_current, y_time_series_current_ptr)[0]
                                print("m_c_coefficients: ", m_c_coefficients)
                            if self.correlation_observations == 0:
                                self.max_correlation_index_buffer = np.append(self.max_correlation_index_buffer, correlation_index)
                                self.max_correlation_index_buffer = np.delete(self.max_correlation_index_buffer, 0, 0)
                            else:
                                self.max_correlation_index_buffer = np.append(self.max_correlation_index_buffer, correlation_index)
                            if self.max_correlation_index_buffer.shape[0] > 3:
                                self.max_correlation_index_buffer = self.max_correlation_index_buffer[-3:]

                            if self.correlation_ind_processing == 1:
                                if self.max_correlation_index_buffer.shape[0] >= 3:
                                    current_avg_buffer = self.max_correlation_index_buffer
                                    average_delay = gmean(current_avg_buffer)
                                    average_delay = np.round(average_delay)
                                    best_index = np.argmin(average_delay)
                                    input_data_freq_normalized = np.dot(np.diag(synch_freq_data_normalized[0]), tiled_delay_matrix[:, int(current_avg_buffer[best_index])])  # -1
                                else:
                                    input_data_freq_normalized = np.dot(np.diag(synch_freq_data_normalized[0]), tiled_delay_matrix[:, correlation_index])
                            else:
                                input_data_freq_normalized = np.dot(np.diag(synch_freq_data_normalized[0]), tiled_delay_matrix[:, correlation_index])

                            h_est1 = np.zeros((int(self.NFFT), 1), dtype=complex)
                            input_data_freq_rotated = (input_data_freq_normalized * np.conj(self.synch_reference)) / (1 + (1 / self.SNR))

                            h_est00 = np.reshape(input_data_freq_rotated, (input_data_freq_rotated.shape[0], self.num_of_synchs_and_synch_bins[0]))
                            h_est0 = h_est00.T

                            channel_estimate_avg_across_synchs = np.sum(h_est0, axis=0) / (self.num_of_synchs_and_synch_bins[0])
                            h_est1[self.synch_used_bins, 0] = channel_estimate_avg_across_synchs

                            self.est_chan_freq_p[antenna_index, self.correlation_observations, 0:len(h_est1)] = h_est1[:, 0]
                            self.est_chan_freq_n[antenna_index, self.correlation_observations, 0:len(channel_estimate_avg_across_synchs)] = channel_estimate_avg_across_synchs

                            h_est_time = np.fft.ifft(h_est1[:, 0], int(self.NFFT))
                            self.est_chan_impulse[antenna_index, self.correlation_observations, 0:len(h_est_time)] = h_est_time

                            h_est_ext = np.tile(channel_estimate_avg_across_synchs, (1, self.num_of_synchs_and_synch_bins[0])).T

                            synch_equalized = (input_data_freq_normalized * np.conj(h_est_ext[:, 0])) / ((np.conj(h_est_ext[:, 0]) * h_est_ext[:, 0]) + (1 / (self.SNR + 1e-10)))
                            self.est_synch_freq[antenna_index, self.correlation_observations, 0:len(self.synch_used_bins) * self.num_of_synchs_and_synch_bins[0]] = synch_equalized

                loop_count += 1

        if self.num_ant_txrx == 1:
            antenna_index = 0
            for corr_obs_index in range(self.correlation_observations):
                for data_symbol_index in range(self.synch_data_pattern[1]):
                    if sum(self.correlation_frame_index_value_buffer[antenna_index, corr_obs_index, :]) + self.NFFT < input_with_freq_offset.shape[0]:
                        data_ptr = int(self.correlation_frame_index_value_buffer[antenna_index, corr_obs_index, 0] + (data_symbol_index + 1) * self.symbol_length)
                        self.rx_buffer_time_data = input_with_freq_offset[data_ptr: data_ptr + self.NFFT]

                        fft_vec = np.fft.fft(self.rx_buffer_time_data, self.NFFT)

                        input_data_freq = fft_vec[self.used_bins_data]
                        input_pow_est = sum(input_data_freq * np.conj(input_data_freq)) / len(input_data_freq)

                        input_data_freq_normalized = input_data_freq / np.sqrt(input_pow_est)
                        channel_estimate_avg_across_synchs = self.est_chan_freq_p[antenna_index, corr_obs_index, self.used_bins_data]

                        del_rotate = np.exp(1j * 2 * (np.pi / self.NFFT) * self.used_bins_data * self.correlation_frame_index_value_buffer[antenna_index, corr_obs_index, 1])
                        input_data_freq_rotated = np.dot(np.diag(input_data_freq_normalized), del_rotate)

                        input_data_freq_equalized = (input_data_freq_rotated * np.conj(channel_estimate_avg_across_synchs)) / (
                                (np.conj(channel_estimate_avg_across_synchs) * channel_estimate_avg_across_synchs) + (1 / self.SNR))

                        if corr_obs_index * self.synch_data_pattern[1] + data_symbol_index == 0:
                            self.est_data_freq[antenna_index, corr_obs_index, :] = self.est_data_freq[antenna_index, corr_obs_index, :] + input_data_freq_equalized
                        else:
                            self.est_data_freq = np.vstack((self.est_data_freq[antenna_index, :], input_data_freq_equalized))
                            self.est_data_freq = self.est_data_freq[np.newaxis, :, :]
                        data = self.est_data_freq[antenna_index, corr_obs_index, 0:len(self.used_bins_data)]
                        p_est1 = sum(data * np.conj(data)) / (len(data) + 1e-10)

                        self.est_data_freq[antenna_index, corr_obs_index * self.synch_data_pattern[1] + data_symbol_index, 0:len(self.used_bins_data)] /= np.sqrt(p_est1)

                        data_out = self.est_data_freq[antenna_index, corr_obs_index * self.synch_data_pattern[1] + data_symbol_index, 0:len(self.used_bins_data)]
                        out[0:len(data_out)] = data_out
        return len(output_items[0])
