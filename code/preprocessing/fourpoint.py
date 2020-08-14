"""MIT License

Copyright (c) 2020 Tuomas Tiainen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

# Numpy license must be included
import numpy as np
from numpy import mean
import pandas as pd
import io
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from math import sin, cos, sqrt
from cmath import exp

import traceback

from mpl_toolkits.mplot3d import Axes3D

from plot_utils import add_scale, polar_plot, barchart
from generate_signals import generate_profile_signals

import warnings
import os

from generate_signals import polar_to_cartesian

warnings.filterwarnings('ignore')

# np.warnings.filterwarnings('ignore')
# np.testing.suppress_warnings()

# SAMPLES_IN_ROUND = 1024
SAMPLES_IN_ROUND = 1024
# SAMPLES_IN_ROUND = 256
# SAMPLES_IN_ROUND = 128
# FILTERLEVEL = 6
FILTERLEVEL = 100    #near highest level that doesn't corrupt data


def import_data_lvm(fn):
    import lvm_read
    data = lvm_read.read(fn)
    data = data[0]["data"]
    signals = [[i[a] for i in data] for a in [1, 2, 3, 4]]
    rot_freq = [i[5] for i in data if i[5]]
    return (signals, mean(rot_freq[0:20]))

def get_avg_freq(speeds):
    speeds = speeds.apply(lambda x: x.str.replace(',','.'))
    speeds = speeds.values[1:20]
    speeds = speeds.astype('float64')
    freq = np.round(np.mean(speeds),6)
    return freq

def extract_data(file_path):
    """Returns (signals,mean_freq,accelerations)"""
    f = open(file_path,'r')
    fl = f.readline()
    while '***End_of_Header***'not in fl:
        fl = f.readline()

    fl = f.readline()
    while '***End_of_Header***'not in fl:
        fl = f.readline()

    Header = f.readline().split('\t')
    fl = f.read()
    dataframe = pd.read_csv(io.StringIO(fl), sep='\t',engine = 'python',names = Header)
    #print(dataframe)
    speeds = dataframe[['Untitled']].copy()
    freq = get_avg_freq(speeds)
    #print(freq)
    signals = dataframe[['laser s1','laser s2','laser s3','laser s4']].copy()
    accelerations = dataframe[['kiihtyvyys x1', 'kiihtyvyys y1', 'kiihtyvyys x2', 'kiihtyvyys y2']].copy()

    signals = signals.apply(lambda x: x.str.replace(',','.')).values.astype('float64')
    accelerations = accelerations.apply(lambda x: x.str.replace(',','.')).values.astype('float64')
    #print(signals.shape)
    #print(accelerations.shape)

    return (signals, freq, accelerations)

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def average(data, remove_first_round=False, rounds=None):
    # remove first round samples
    if remove_first_round:
        data = [signal[SAMPLES_IN_ROUND:] for signal in data]

    if rounds:
        start = int(SAMPLES_IN_ROUND * rounds[0])
        print(start)
        end = int(SAMPLES_IN_ROUND * rounds[1])
        print(end)
        data = [signal[start:end] for signal in data]

    #remove mean

    for index, signal in enumerate(data):
        s_mean = mean(signal)
        data[index] = np.array(signal) - s_mean

    rounds = int(len(data[0]) / SAMPLES_IN_ROUND)

    # l = 8
    # data = [i[l * 1024:(l + 1) * 1024] for i in data]

    for index, signal in enumerate(data):
        s_chunks = list(chunks(signal, SAMPLES_IN_ROUND))
        signal = [mean([val[i] for val in s_chunks]) for i in range(SAMPLES_IN_ROUND)]
        data[index] = signal

    # if invert:
    # if True:
    #     data = [signal * -1 for signal in data]

    # this data contains SAMPLES_IN_ROUND points averaged over each round
    return [np.array(s) for s in data]


def ozono_f_coeff(signals, angles):
    a_1, a_2, a_3 = angles[0], angles[1], angles[2]

    a_1 = (angles[0] / 360) * (2 * np.pi)
    a_2 = (angles[1] / 360) * (2 * np.pi)
    a_3 = (angles[2] / 360) * (2 * np.pi)

    def equations(p):
        w_2, w_3 = p
        return (sin(a_1) + w_2 * sin(a_2) + w_3 * sin(a_3), cos(a_1) + w_2 * cos(a_2) + w_3 * cos(a_3))

    w_2, w_3 = fsolve(equations, (1, 1))

    arr = np.array([signals[0], w_2 * signals[1], w_3 * signals[2]])
    s = arr.sum(axis=0)

    fft_s = np.fft.fft(s, SAMPLES_IN_ROUND)

    ozono_coefficients = [0 + 0J] * SAMPLES_IN_ROUND

    for k, fft in enumerate(fft_s):
        alpha_k = (cos(k * a_1) + w_2 * cos(k * a_2) + w_3 * cos(k * a_3))
        beta_k = (sin(k * a_1) + w_2 * sin(k * a_2) + w_3 * sin(k * a_3))

        C_k = (2 * fft).real
        D_k = (2 * fft).imag

        A_k = (alpha_k * C_k - beta_k * D_k) / (alpha_k**2 + beta_k**2)
        B_k = (beta_k * C_k + alpha_k * D_k) / (alpha_k**2 + beta_k**2)

        ozono_coefficients[k] = (A_k + B_k * 1J) / 2

    coeff = np.array(ozono_coefficients)
    ecc = get_ecc(signals, angles)
    coeff[1] = ecc
    return coeff


def diameter_f_coeff(signals):
    # d1 = np.add(signals[0], signals[1])
    # print(len(d1))

    # d2 = d1[int(len(d1) / 2):-1] +  d1[0:int(len(d1) / 2)]
    # print(len(d2))

    # deltar = 0.25 * np.add(d1, d2)

    deltar = 0.5 * np.add(signals[0], signals[1])
    fft = np.fft.fft(deltar, SAMPLES_IN_ROUND)
    return fft


def hybrid_merge(diameter, ozono):
    array = [0] * SAMPLES_IN_ROUND

    for index, item in enumerate(array):
        if (index % 2) == 0:
            array[index] = diameter[index]
        else:
            array[index] = ozono[index]

    return np.array(array)


def filter_fft(fft, filterlevel=FILTERLEVEL, include_ecc=False):
    fft = list(fft)

    # pad with zeros
    fft += ([0] * int(((SAMPLES_IN_ROUND / 2) - len(fft))))
    fft = np.array(fft)

    # (int(SAMPLES_IN_ROUND / 2)

    fft = fft[0:(int(SAMPLES_IN_ROUND / 2))]
    fft[0] = 0

    if not include_ecc:
        fft[1] = 0

    fft[filterlevel + 1:] = 0

    # the fourier is symmetric, use only first half, then mirror complex conjugates excluding first
    fft = np.append(fft, np.append([0], np.conj(fft[1:][::-1])))

    return fft


def get_roundness_profile(fft, samples=SAMPLES_IN_ROUND):
    theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    r = np.fft.ifft(fft).real
    roundness = max(r) - min(r)
    return (roundness, theta, r)


def hybrid_f_coeff(signals, angles=None):

    if not angles:
        angles = [0, 38, 67]

    ozono = ozono_f_coeff([signals[0], signals[1], signals[2]], angles)
    diameter = diameter_f_coeff([signals[0], signals[3]])
    hybrid_fourpoint = hybrid_merge(diameter, ozono)
    return hybrid_fourpoint


def get_matrix_f(angles):
    def matrix(n):
        m = []
        for a in angles:
            m.append(np.array([cos(a), sin(a), exp(-1j * (n * a))]))

        return np.array(m)

    return matrix


def tiainen_roundness(signals, angles_deg, filterlevel=FILTERLEVEL):
    if len(angles_deg) > len(signals):
        raise "Error: No signal for each angle."

    angles = [((i / 360) * 2 * np.pi) for i in angles_deg]

    f = get_matrix_f(angles)
    # [plt.plot(s) for s in signals]
    fft_coefficients = [np.fft.fft(s, SAMPLES_IN_ROUND) for s in signals[0:len(angles)]]
    # [plt.plot(fft_coefficient) for fft_coefficient in fft_coefficients]
    # plt.show()

    harmonics = np.arange(1, filterlevel + 1)
    r = []

    for harmonic in harmonics:
        h = f(harmonic)
        s_fourier_coefficients = [np.array(fft_coefficient[harmonic]) for fft_coefficient in fft_coefficients]

        ht = np.matrix.transpose(h)
        hth = np.matmul(ht, h)
        try:
            x_estimate = np.matmul(np.matmul(np.linalg.inv(hth), ht), s_fourier_coefficients)
        except np.linalg.linalg.LinAlgError:
            # print(traceback.format_exc())
            x_estimate = [0 + 0J] * 3

        r.append(x_estimate[2])

    fft = np.array([0 + 0J] * SAMPLES_IN_ROUND)

    for index, value in enumerate(r):
        fft[index + 1] = value

    fft = fft[0:(int(SAMPLES_IN_ROUND / 2))]
    fft[0] = 0 + 0J
    fft[1] = 0 + 0J
    fft[filterlevel + 1:] = 0
    fft = np.append(fft, np.append([0], np.conj(fft[1:][::-1])))

    return fft


def advanced_roundness_f_coeff(signals, angles_deg, filterlevel=FILTERLEVEL):
    if len(angles_deg) > len(signals):
        raise "Error: No signal for each angle."

    angles = [((i / 360) * 2 * np.pi) for i in angles_deg]

    f = get_matrix_f(angles)
    # [plt.plot(s) for s in signals]
    fft_coefficients = [np.fft.fft(s, SAMPLES_IN_ROUND) for s in signals[0:len(angles)]]
    # [plt.plot(fft_coefficient) for fft_coefficient in fft_coefficients]
    # plt.show()

    harmonics = np.arange(1, filterlevel + 1)
    r = []

    for harmonic in harmonics:
        h = f(harmonic)
        s_fourier_coefficients = [np.array(fft_coefficient[harmonic]) for fft_coefficient in fft_coefficients]

        ht = np.matrix.transpose(h)
        hth = np.matmul(ht, h)
        try:
            x_estimate = np.matmul(np.matmul(np.linalg.inv(hth), ht), s_fourier_coefficients)
        except np.linalg.linalg.LinAlgError:
            # print(traceback.format_exc())
            x_estimate = [0 + 0J] * 3

        r.append(x_estimate[2])

    fft = np.array([0 + 0J] * SAMPLES_IN_ROUND)

    for index, value in enumerate(r):
        fft[index + 1] = value

    fft = fft[0:(int(SAMPLES_IN_ROUND / 2))]
    fft[0] = 0 + 0J
    # fft[1] = 0 + 0J
    fft[filterlevel + 1:] = 0
    fft = np.append(fft, np.append([0], np.conj(fft[1:][::-1])))

    return fft


def plots_real_data():
    data = import_data_lvm("matlab/4.00Hz_300mm.lvm")
    signals = np.array(average(data[0]))

    advanced = advanced_roundness_f_coeff(signals, [0, 38, 67])
    advanced2 = advanced_roundness_f_coeff(signals, [0, 38, 67, 180])
    ozono = ozono_f_coeff(signals, [0, 38, 67])
    hybrid = hybrid_f_coeff(signals, angles=[0, 38, 67])

    filtered_ozono = filter_fft(ozono)
    filtered_hybrid = filter_fft(hybrid)

    barchart(filter_fft(hybrid, FILTERLEVEL, True), FILTERLEVEL)
    plt.show()

    ar = get_roundness_profile(advanced)
    ar2 = get_roundness_profile(advanced2)
    o = get_roundness_profile(filtered_ozono)
    h = get_roundness_profile(filtered_hybrid)

    polar_plot(o, "Ozono", 0.05)
    polar_plot(h, "Hybrid four point", 0.05)
    polar_plot(ar, "Advanced roundness (3 probes)", 0.05)
    polar_plot(ar2, "Advanced roundness (4 probes)", 0.05)
    plt.show()


def plots_generated():
    # signals = np.array(average(SIGNALS))

    X = []
    e_ar = []
    e_ar2 = []
    e_ozono = []
    e_hybrid = []
    diam = 100

    l = 0
    h = 10
    for ecc in np.linspace(l, h, 11):

        signals, profile = generate_profile_signals(ecc=(ecc / 100) * diam)
        signals = np.array(average(signals))

        advanced = advanced_roundness_f_coeff(signals, [0, 38, 67])
        advanced2 = advanced_roundness_f_coeff(signals, [0, 38, 67, 180])
        ozono = ozono_f_coeff(signals, [0, 38, 67])
        hybrid = hybrid_f_coeff(signals, angles=[0, 38, 67])

        filtered_ozono = filter_fft(ozono)
        filtered_hybrid = filter_fft(hybrid)

        ar = get_roundness_profile(advanced)
        ar2 = get_roundness_profile(advanced2)
        o = get_roundness_profile(filtered_ozono)
        h = get_roundness_profile(filtered_hybrid)

        X.append(ecc / diam)

        # e_ar.append(RRMSE(profile, ar))
        # e_ar2.append(RRMSE(profile, ar2))
        # e_ozono.append(RRMSE(profile, o))
        # e_hybrid.append(RRMSE(profile, h))

        e_ar.append(RPPE(profile, ar))
        e_ar2.append(RPPE(profile, ar2))
        e_ozono.append(RPPE(profile, o))
        e_hybrid.append(RPPE(profile, h))

        polar_plot(profile, "Actual profile", True, title="Roundness profile e={}% of d".format(ecc))
        polar_plot(o, "Ozono", False, title="Roundness profile e={}% of d".format(ecc))
        polar_plot(h, "Hybrid four point", False, title="Roundness profile e={}% of d".format(ecc))
        polar_plot(ar, "Advanced roundness (3 probes)", False, title="Roundness profile e={}% of d".format(ecc))
        polar_plot(ar2, "Advanced roundness (4 probes)", False, title="Roundness profile e={}% of d".format(ecc))
        plt.show()
        # plt.savefig("profile_{}.png".format(ecc))
        # plt.clf()

    # plt.plot(X, e_ar, label="ar")
    plt.plot(X, e_ar2, label="ar2")
    plt.plot(X, e_ozono, label="ozono")
    plt.plot(X, e_hybrid, label="hybrid")

    plt.legend()

    plt.show()


def RRMSE(S, S_e):
    S = S[2]
    S_e = S_e[2]

    squaresum = []

    for index, value in enumerate(S):
        squaresum.append((S[index] - S_e[index])**2)

    squaresum = sum(squaresum)

    return 1 / ((max(S) - min(S)) * np.sqrt((1 / len(S)) * squaresum))


def RPPE(S, S_e):
    S = S[2]
    S_e = S_e[2]
    return abs(max(S) - min(S) - (max(S_e) - min(S_e))) / (max(S) - min(S_e))


def phase_diff():
    # data = import_data_lvm("matlab/4.00Hz_300mm.lvm")
    # data = import_data_lvm("matlab/4.18Hz_1150mm.lvm")
    data = import_data_lvm("matlab/4.38Hz_1150mm.lvm")
    # data = import_data_lvm("matlab/4.77Hz_1150mm.lvm")
    signals = data[0]

    # signals, profile = generate_profile_signals(ecc=0)

    signals = np.array(average(signals, False))

    s1, s2, s3, s4 = signals[0], signals[1], signals[2], signals[3]

    # print(signals)
    # s1 = [i + (0.08 - max(s1)) for i in s1]
    # s2 = [i + (0.08 - max(s2)) for i in s2]
    # s3 = [i + (0.08 - max(s3)) for i in s3]
    # s4 = [i + (0.08 - max(s4)) for i in s4]

    x = np.arange(0, len(signals[0]))
    plt.plot(x, s1, label="S1")
    plt.plot(x, s2, label="S2")
    plt.plot(x, s3, label="S3")
    plt.plot(x, s4, label="S4")
    plt.legend()
    plt.show()

    xcor2 = np.correlate(s1, s2, "full")
    lag = np.argmax(xcor2) - (xcor2.size + 1) / 2
    # print(lag)
    print(lag / 1024 * 360)

    xcor3 = np.correlate(s1, s3, "full")
    lag = np.argmax(xcor3) - (xcor3.size + 1) / 2
    # print(lag)
    print(lag / 1024 * 360)

    xcor4 = np.correlate(s1, s4, "full")
    lag = np.argmax(xcor4) - (xcor4.size + 1) / 2
    # print(lag)
    print(lag / 1024 * 360)

    x = np.arange(0, len(xcor2))
    plt.plot(x, xcor2, label="S1xS2")
    plt.plot(x, xcor3, label="S1xS3")
    plt.plot(x, xcor4, label="S1xS4")
    plt.axvline(np.argmax(xcor2), linewidth=3, color='r', ls='dashed')
    plt.axvline(np.argmax(xcor3), linewidth=3, color='r', ls='dashed')
    plt.axvline(np.argmax(xcor4), linewidth=3, color='r', ls='dashed')
    plt.legend()
    plt.show()


def remove_radius(profile):
    return [profile[0], profile[1], [i - profile[0] for i in profile[2]]]


def get_ecc(signals, angles):
    #  eccentricity
    #  Eccentricity as average of 1H terms.
    #  Fixed phase shift in frequency domain.
    angles = [exp(-1j * np.deg2rad(i)) for i in angles]

    phase_shift = [1] + angles[1:]
    fft_s = [np.fft.fft(s, SAMPLES_IN_ROUND)[1] for s in signals[0:len(angles)]]
    ecc = np.mean(np.multiply(fft_s, phase_shift))
    return ecc


def plot_generated():

    # cpm_points = r_cpm(1.00)
    # cpm_points = circle_cpm(1.20)
    errors = False

    if errors:
        s = nr(0.1, 0.03)
        cpm_points, cpm_data = r_cpm(s)
        angles_deg_errors = [nr(0, 0.5) for i in range(4)]
        vertical_error = nr(0, 0.35)
        horizontal_error = nr(0, 0.25)
    else:
        s = 0
        cpm_points, cpm_data = r_cpm(s)
        angles_deg_errors = [0 for i in range(4)]
        vertical_error = 0
        horizontal_error = 0

    signals, profile = generate_profile_signals(
        SAMPLES_IN_ROUND,
        cpm_points=cpm_points,
        vertical_error=vertical_error,
        horizontal_error=horizontal_error,
        angles_deg_errors=angles_deg_errors)

    # probe scale error
    if errors:
        signals = [[i + nr(0, 0.003) for i in signal] for signal in signals]

    signals = np.array(average(signals))
    profile = remove_radius(profile)

    signals = [s * -1 for s in signals]

    advanced = advanced_roundness_f_coeff(signals, [0, 38, 67])
    advanced2 = advanced_roundness_f_coeff(signals, [0, 38, 67, 180])
    ozono = ozono_f_coeff(signals, [0, 38, 67])
    hybrid = hybrid_f_coeff(signals, angles=[0, 38, 67])

    filtered_ozono = filter_fft(ozono, include_ecc=False)
    filtered_hybrid = filter_fft(hybrid, include_ecc=False)

    filtered_advanced = filter_fft(advanced, include_ecc=False)
    filtered_advanced2 = filter_fft(advanced2, include_ecc=False)

    # print(ecc)
    # print(np.abs(ecc))
    # print(np.angle(ecc))

    barchart(filtered_ozono)
    plt.show()
    # barchart(advanced)
    # plt.show()

    ar = get_roundness_profile(filtered_advanced)
    ar2 = get_roundness_profile(filtered_advanced2)
    o = get_roundness_profile(filtered_ozono)
    h = get_roundness_profile(filtered_hybrid)

    offset = 0.30
    polar_plot(profile, "Actual profile", offset, True)
    # polar_plot(profile, "Actual profile", offset)
    polar_plot(o, "Ozono", offset)
    polar_plot(h, "Hybrid four point", offset)
    polar_plot(ar, "Advanced roundness (3 probes)", offset)
    polar_plot(ar2, "Advanced roundness (4 probes)", offset)
    plt.show()


def get_cpm(signals, angles, roundness_profile_f):
    s1, s2 = signals
    a1, a2 = np.deg2rad(angles)

    def get_k(datalen):
        arr_1 = np.arange(0, 1 + datalen / 2)
        arr_2 = np.flip(np.arange(1, datalen / 2) * -1)
        return (np.concatenate([arr_1, arr_2]))

    def shift_fd(profile, angle):
        k = get_k(SAMPLES_IN_ROUND)
        k = k * angle * -1j
        return profile * np.exp(k)
        return np.multiply(profile, np.exp(k))

    def shift_r(profile, angle):
        return np.roll(profile, int(SAMPLES_IN_ROUND * angle / 2 * np.pi))

    rprof = get_roundness_profile(roundness_profile_f)[2]
    shifted_to_s1 = get_roundness_profile(shift_fd(roundness_profile_f, a1))[2]
    cpm_x_raw = signals[0] - shifted_to_s1

    shifted_to_s2 = get_roundness_profile(shift_fd(roundness_profile_f, a2))[2]
    cpm_y_raw = (s2 - shifted_to_s2 - cpm_x_raw * np.cos(a2)) / np.sin(a2)

    fft_cpm_x = filter_fft(np.fft.fft(cpm_x_raw), filterlevel=FILTERLEVEL, include_ecc=True)
    fft_cpm_y = filter_fft(np.fft.fft(cpm_y_raw), filterlevel=FILTERLEVEL, include_ecc=True)

    cpm_x = np.fft.ifft(fft_cpm_x)
    cpm_y = np.fft.ifft(fft_cpm_y)

    return np.array([cpm_x * -1, cpm_y * -1])

def harmonic_cpm(amplitude=0.0, harmonic=1, count=SAMPLES_IN_ROUND):
    t = np.linspace(0, 2 * np.pi, count)

    # cpm_points = polar_to_cartesian(t, [0.01] * 1024)
    # return cpm_points

    def rp():
        return np.random.random() * np.pi * 2

    def rd_cpm():
        rps = [rp() for i in range(8)]
        return np.array([(np.sin(harmonic * i + rps[1]) + 0.8 * np.sin(2 * i + rps[2]) + 0.6 * np.sin(3 * i + rps[3]) +
                          0.4 * np.sin(4 * i + rps[4]) + 0.2 * np.sin(5 * i + rps[5])) for i in t])

    cpm_x = rd_cpm() * amplitude
    cpm_y = cpm_x
    # print(cpm_x)
    # print(cpm_y)

    return list(zip(cpm_x, cpm_y))


def r_cpm(c=0.0, count=SAMPLES_IN_ROUND):
    t = np.linspace(0, 2 * np.pi, count)

    # cpm_points = polar_to_cartesian(t, [0.01] * 1024)
    # return cpm_points

    def rp():
        return np.random.random() * np.pi * 2

    def rd_cpm():
        rps = [rp() for i in range(5)]
        cpm = np.array([(np.sin(1 * i + rps[0]) + 0.8 * np.sin(2 * i + rps[1]) + 0.6 * np.sin(3 * i + rps[2]) +
                         0.4 * np.sin(4 * i + rps[3]) + 0.2 * np.sin(5 * i + rps[4])) for i in t])
        return [cpm, rps]

    rd = rd_cpm()
    cpm_x_points, cpm_x_phases = rd[0], rd[1]
    cpm_x = cpm_x_points * c

    rd = rd_cpm()
    cpm_y_points, cpm_y_phases = rd[0], rd[1]
    cpm_y = cpm_y_points * c

    return list(zip(cpm_x, cpm_y)), {"cpm_c": c, "cpm_x_phases": cpm_x_phases, "cpm_y_phases": cpm_y_phases}


def circle_cpm(amplitude=0.0, count=SAMPLES_IN_ROUND):
    t = np.linspace(0, 2 * np.pi, count)
    cpm_points = polar_to_cartesian(t, [amplitude] * SAMPLES_IN_ROUND)
    return cpm_points


def cpm_generated():
    s = 0.1
    cpm_points, cpm_data = r_cpm(s)
    gen_cpm = [[i[0] for i in cpm_points], [i[1] for i in cpm_points]]
    plot_cpms([gen_cpm], ["Center point movement"])
    plt.show()
    return
    # cpm_points = circle_cpm(104.20)

    signals, profile = generate_profile_signals(SAMPLES_IN_ROUND, cpm_points=cpm_points)
    signals = np.array(average(signals))

    advanced = advanced_roundness_f_coeff(signals, [0, 38, 67])
    advanced2 = advanced_roundness_f_coeff(signals, [0, 38, 67, 180])
    ozono = ozono_f_coeff(signals, [0, 38, 67])
    hybrid = hybrid_f_coeff(signals, angles=[0, 38, 67])

    filtered_advanced = filter_fft(advanced)
    filtered_advanced2 = filter_fft(advanced2)
    filtered_ozono = filter_fft(ozono)
    filtered_hybrid = filter_fft(hybrid)

    s = [signals[0], signals[2]]
    angles = [0, 67]

    ozono_cpm = get_cpm(s, angles, filtered_ozono)
    hybrid_cpm = get_cpm(s, angles, filtered_hybrid)
    ar_cpm = get_cpm(s, angles, filtered_advanced)
    ar2_cpm = get_cpm(s, angles, filtered_advanced2)
    plot_cpms([gen_cpm, ozono_cpm, hybrid_cpm, ar_cpm, ar2_cpm],
              ["Generated", "Ozono", "Jansen (3 probes)", "Jansen (4 probes)", "Hybrid"])

    plt.show()


def cpm_real_data():
    data = import_data_lvm("matlab/4.00Hz_300mm.lvm")
    # signals = np.array(average(data[0], remove_first_round=False, rounds=[15, 16]))
    signals = np.array(average(data[0]))

    advanced = advanced_roundness_f_coeff(signals, [0, 38, 67])
    advanced2 = advanced_roundness_f_coeff(signals, [0, 38, 67, 180])
    ozono = ozono_f_coeff(signals, [0, 38, 67])
    hybrid = hybrid_f_coeff(signals, angles=[0, 38, 67])

    filtered_advanced = filter_fft(advanced)
    filtered_advanced2 = filter_fft(advanced2)
    filtered_ozono = filter_fft(ozono)
    filtered_hybrid = filter_fft(hybrid)

    s = [signals[0], signals[2]]
    angles = [0, 67]

    # ar = get_roundness_profile(advanced)
    ozono_cpm = get_cpm(s, angles, filtered_ozono)
    hybrid_cpm = get_cpm(s, angles, filtered_hybrid)
    ar_cpm = get_cpm(s, angles, filtered_advanced)
    ar2_cpm = get_cpm(s, angles, filtered_advanced2)
    plot_cpms([ozono_cpm, hybrid_cpm, ar_cpm, ar2_cpm], ["Ozono", "Hybrid", "Jansen (3 probe)", "Jansen (4 probe)"])

    plt.show()



def cpm_real_data_2(file_path):

    data = extract_data(file_path)
    # signals = np.array(average(data[0], remove_first_round=False, rounds=[15, 16]))
    signals_array = np.array(data[0])
    accelerations = np.array(data[2])
    dpoints = int(signals_array[:,0].shape[0])
    rounds = int(dpoints/SAMPLES_IN_ROUND)
    #somethings = average(signals_array.T,rounds = [0,1])
    signals_array = signals_array.T
    accelerations = accelerations
    cpm_array = np.zeros((dpoints,2))
    ozono_arr = np.zeros((2,dpoints))
    hybrid_arr = np.zeros((2,dpoints))
    ar_arr = np.zeros((2,dpoints))
    ar2_arr = np.zeros((2,dpoints))

    for round in range(rounds):
        start = round*SAMPLES_IN_ROUND
        #print("start: {}".format(start))
        stop = start + SAMPLES_IN_ROUND
        #print("stop: {}".format(stop))
        signals = signals_array[:,start:stop]
        advanced = advanced_roundness_f_coeff(signals, [0, 38, 67])
        advanced2 = advanced_roundness_f_coeff(signals, [0, 38, 67, 180])
        ozono = ozono_f_coeff(signals, [0, 38, 67])
        hybrid = hybrid_f_coeff(signals, angles=[0, 36.91, 66.45])

        filtered_advanced = filter_fft(advanced)
        filtered_advanced2 = filter_fft(advanced2)
        filtered_ozono = filter_fft(ozono)
        filtered_hybrid = filter_fft(hybrid)
        polar_plot(get_roundness_profile(filtered_hybrid),"Hybrid roundness of one round",0.05)

        s = [signals[0], signals[2]]
        angles = [0, 66.45]

        # ar = get_roundness_profile(advanced)
        ozono_cpm = get_cpm(s, angles, filtered_ozono)
        hybrid_cpm = get_cpm(s, angles, filtered_hybrid)
        ar_cpm = get_cpm(s, angles, filtered_advanced)
        ar2_cpm = get_cpm(s, angles, filtered_advanced2)

        #print(hybrid_cpm.T.shape)

        cpm_array[start:stop,0] = hybrid_cpm.T[:,0]
        cpm_array[start:stop,1] = hybrid_cpm.T[:,1]

        ozono_arr[0,start:stop]=ozono_cpm[0,:]
        ozono_arr[1,start:stop]=ozono_cpm[1,:]

        hybrid_arr[0,start:stop]=hybrid_cpm[0,:]
        hybrid_arr[1,start:stop]=hybrid_cpm[1,:]

        ar_arr[0,start:stop]=ar_cpm[0,:]
        ar_arr[1,start:stop]=ar_cpm[1,:]

        ar2_arr[0,start:stop]=ar2_cpm[0,:]
        ar2_arr[1,start:stop]=ar2_cpm[1,:]

    #plot_cpms([ozono_arr[:,:2048], hybrid_arr[:,:2048], ar_arr[:,:2048], ar2_arr[:,:2048]], ["Ozono", "Hybrid", "Jansen (3 probe)", "Jansen (4 probe)"])

    #plt.show()
    data_arr = np.concatenate((cpm_array,accelerations),axis = 1)
    print(data_arr.shape)
    return data_arr, data[1]

def plot_cpms(cpms, labels=None):

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    # ax = plt.subplot(projection='3d')

    for cpm, label in zip(cpms, labels):
        x = cpm[0]
        z = cpm[1]
        y = list(np.arange(len(x)))

        ax.plot(x, y, z, label=label)
        ax.set_zlim(ax.get_xlim())

        ax.set_xlabel("x")
        ax.set_ylabel("n")
        ax.set_zlabel("y")

        show_path = True

        # if show_path:
        #     ax.view_init(elev=0, azim=-90)

        ax.legend()


import json
from pprint import pprint
import uuid
import pickle


def nr(mu, sigma):
    # normally distributed random number
    # mean and standard deviation
    return np.random.normal(mu, sigma, 1)[0]


def montecarlo_single(n=0):
    print(n)

    if not os.path.exists("res/"):
        os.makedirs("res/")

    errors = False

    if errors:
        mu = 10 * 0.1
        sigma = 10 * 0.03

        s = nr(mu, sigma)
        cpm_points, cpm_data = r_cpm(s)
        angles_deg_errors = [nr(0, 0.25) for i in range(4)]
        vertical_error = nr(0, 0.35)
        horizontal_error = nr(0, 0.25)

    else:
        mu = 0
        sigma = 0
        cpm_points, cpm_data = r_cpm(0)
        angles_deg_errors = [0 for i in range(4)]
        vertical_error = 0
        horizontal_error = 0

    signals, profile = generate_profile_signals(
        SAMPLES_IN_ROUND,
        cpm_points=cpm_points,
        vertical_error=vertical_error,
        horizontal_error=horizontal_error,
        angles_deg_errors=angles_deg_errors)

    # add probe scaling error

    signals = np.array(average(signals))

    advanced = advanced_roundness_f_coeff(signals, [0, 38, 67])
    advanced2 = advanced_roundness_f_coeff(signals, [0, 38, 67, 180])
    ozono = ozono_f_coeff(signals, [0, 38, 67])
    hybrid = hybrid_f_coeff(signals, angles=[0, 38, 67])

    filtered_ozono = filter_fft(ozono, include_ecc=True)
    filtered_hybrid = filter_fft(hybrid, include_ecc=True)
    filtered_advanced = filter_fft(advanced, include_ecc=True)
    filtered_advanced2 = filter_fft(advanced2, include_ecc=True)

    errors = {
        "SAMPLES_IN_ROUND": SAMPLES_IN_ROUND,
        "cpm_data": cpm_data,
        "vertical_error": vertical_error,
        "horizontal_error": horizontal_error,
        "angles_deg_errors": angles_deg_errors
    }

    res = {
        "ozono": list(filtered_ozono[0:FILTERLEVEL]),
        "hybrid": list(filtered_hybrid[0:FILTERLEVEL]),
        "advanced": list(filtered_advanced[0:FILTERLEVEL]),
        "advanced2": list(filtered_advanced2[0:FILTERLEVEL])
    }
    d = {"res": res, "errors": errors}
    if True:

        with open("res/res-mu{}-sigma{}-{}.pickle".format(mu, sigma, uuid.uuid4()), "wb") as f:
            f.write(pickle.dumps(d))

    # barchart(filter_fft(advanced, include_ecc=False))
    # plt.show()
    # barchart(filter_fft(ozono, include_ecc=False))
    # plt.show()
    # barchart(ozono)
    # barchart(advanced)
    # plt.show()
    # barchart(ozono)
    # plt.show()
    # ar = get_roundness_profile(filtered_advanced)
    # ar2 = get_roundness_profile(filtered_advanced2)
    # o = get_roundness_profile(filtered_ozono)
    # h = get_roundness_profile(filtered_hybrid)

    # polar_plot(o, "Ozono", 0.05)
    # polar_plot(h, "Hybrid four point", 0.05)
    # polar_plot(ar, "Advanced roundness (3 probes)", 0.05)
    # polar_plot(ar2, "Advanced roundness (4 probes)", 0.05)
    # plt.show()


def plot_incorrectangles():
    s = 0
    cpm_points, cpm_data = r_cpm(s)
    angles_deg_errors = [0 for i in range(4)]
    vertical_error = 0
    horizontal_error = 0

    signals, profile = generate_profile_signals(
        SAMPLES_IN_ROUND,
        cpm_points=cpm_points,
        vertical_error=vertical_error,
        horizontal_error=horizontal_error,
        angles_deg_errors=angles_deg_errors)

    signals = np.array(average(signals))
    profile = remove_radius(profile)

    signals = [s * -1 for s in signals]

    # advanced = advanced_roundness_f_coeff(signals, [0, 38, 67])
    # advanced1 = advanced_roundness_f_coeff(signals, [0, 37.06875, 65.88984375, 179.41640625])
    advanced1 = advanced_roundness_f_coeff(signals, [0, 37.06875, 65.88984375, 179.41640625])
    advanced2 = advanced_roundness_f_coeff(signals, [0, 38, 67, 180])

    filtered_advanced1 = filter_fft(advanced1, include_ecc=False)
    filtered_advanced2 = filter_fft(advanced2, include_ecc=False)

    h = get_roundness_profile(filtered_advanced1)
    h2 = get_roundness_profile(filtered_advanced2)

    offset = 0.45
    polar_plot(profile, "Actual profile", offset)
    polar_plot(h2, "Angles with errors", offset)
    polar_plot(h, "Determined angles", offset)
    plt.show()


def plot_real_data_incorrect():
    import lvm_read
    fn = "./data/teippimittaus_uusi.lvm"

    data = lvm_read.read(fn)

    channels = (data[0]["Channel names"])
    channel_data = data[0]["data"]

    channel_names = ['Laser 1', 'Laser 2', 'Laser 3', 'Laser 4']

    channel_indexes = [channels.index(name) for name in channel_names]

    rounds = [0, 100]
    signals = [[i[a] for i in channel_data][1024 * rounds[0]:1024 * rounds[1]] for a in channel_indexes]

    signals = np.array(average(signals))


    data = import_data_lvm("matlab/4.00Hz_300mm.lvm")
    signals = np.array(average(data[0]))

    advanced1 = advanced_roundness_f_coeff(signals, [0, 38, 67, 180])
    advanced2 = advanced_roundness_f_coeff(signals, [0, 37.06875, 65.88984375, 179.41640625])

    filtered_advanced1 = filter_fft(advanced1)
    filtered_advanced2 = filter_fft(advanced2)

    ar1 = get_roundness_profile(advanced1)
    ar2 = get_roundness_profile(advanced2)

    polar_plot(ar1, "Angles with errors", 0.03)
    polar_plot(ar2, "Determined angles", 0.03)
    plt.show()


if __name__ == "__main__":
    # phase_diff()
    # plots_generated()
    # plots_real_data()
    cpm_real_data()
    # cpm_generated()
    # montecarlo_single()
    #
    # [-37.06875, -65.88984375, -179.41640625]
    # plot_generated()
    # plot_incorrectangles()

    #plot_real_data_incorrect()
