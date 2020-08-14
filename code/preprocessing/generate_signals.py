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

import matplotlib.pyplot as plt
import numpy as np

from shapely import geometry
from shapely.geometry import Point, LinearRing
from shapely.geometry.polygon import Polygon
from shapely import affinity
# from descartes import PolygonPatch

import random
import time

from shapely.ops import cascaded_union

ACCURACY = 5500


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def get_xy(theta_r):
    theta, r = theta_r
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x, y)


def polar_to_cartesian(theta, r):
    return [get_xy(theta_r) for theta_r in zip(theta, r)]


def generate_signals(polygon,
                     angles_deg,
                     samples=1024,
                     cpm_points=None,
                     vertical_error=None,
                     horizontal_error=None,
                     angles_deg_errors=None):

    if not vertical_error:
        vertical_error = 0
    if not horizontal_error:
        horizontal_error = 0

    if angles_deg_errors:
        for index, angle in enumerate(angles_deg):
            angles_deg[index] += angles_deg_errors[index]

    angles = [((i / 360) * 2 * np.pi) for i in angles_deg]

    # TODO: replace 100 with max distance
    probe_distance = 1000
    probe_pos = [Point(probe_distance * np.cos(angle), probe_distance * np.sin(angle)) for angle in angles]
    probe_lines = [geometry.LineString([Point(0, 0), pos]) for pos in probe_pos]

    thetas = np.linspace(0, np.pi * 2, samples)
    # initialize list of signals
    signals = [[] for _ in range(len(angles))]

    # print("starting signals")

    for theta_index, theta in enumerate(thetas):

        # CPM movement
        translated = affinity.translate(polygon, cpm_points[theta_index][0], cpm_points[theta_index][1])

        # error position of the frame
        translated = affinity.translate(translated, horizontal_error, vertical_error)

        rotated = affinity.rotate(translated, theta, use_radians=True)
        for index, angle in enumerate(angles):
            line = probe_lines[index]

            # intersection = line.intersection(rotated)
            intersection = line.intersection(rotated.boundary)
            dist = intersection.distance(probe_pos[index])
            signals[index].append(dist)

    return signals


def plot_polygon(polygon):

    fig = plt.figure(1, figsize=(5, 5), dpi=90)
    ax = fig.add_subplot(111)
    # ring_patch = PolygonPatch(polygon)
    # ax.add_patch(ring_patch)

    # ax.set_title('Profile')
    # xrange = [-2, 2]
    # yrange = [-2, 2]
    # ax.set_xlim(*xrange)
    # ax.set_ylim(*yrange)
    # ax.set_aspect(1)

    # plt.show()


def generate_profile(profile_points, radius):
    theta = np.linspace(0, 2 * np.pi, profile_points)

    r = [radius] * profile_points
    c = 0.01
    # r = [
    #     i[0] + c * np.sin(2 * i[1] + np.pi) + c * np.sin(3 * i[1] + np.pi) + c * np.sin(4 * i[1]) +
    #     c * np.sin(5 * i[1]) + c * np.sin(6 * i[1]) + c * np.sin(7 * i[1]) + c * np.sin(8 * i[1]) +
    #     c * np.sin(9 * i[1]) + c * np.sin(10 * i[1]) for i in zip(r, theta)
    # ]

    # r = [
    #     i[0] + c * np.sin(2 * i[1] + np.pi) + c * np.sin(3 * i[1] + np.pi) + c * np.sin(4 * i[1]) +
    #     c * np.sin(5 * i[1]) + c * np.sin(6 * i[1]) + c * np.sin(7 * i[1]) + c * np.sin(8 * i[1]) +
    #     c * np.sin(9 * i[1]) + c * np.sin(10 * i[1]) for i in zip(r, theta)
    # ]

    phase_deg = [
        0, 0, 9.0, 200.0, 253.0, 20.0, 55.0, 2.0, 137.0, 1.0, 161.0, 212.0, 24.0, 200.0, 245.0, 188.0, 16.0, 11.0,
        75.0, 211.0, 294.0, 165.0, 81.0, 104.0, 59.0, 91.0, 43.65, 157.0, 91.0, 321.0, 161.0
    ]
    phases = [np.deg2rad(phase) for phase in phase_deg]

    r = [i[0] + c * np.sin(3 * i[1]) + c * np.sin(4 * i[1]) for i in zip(r, theta)]

    return theta, r


def create_unit_complex(phase):
    return np.cos(phase) + np.sin(phase) * 1j


def generate_reference_profile_100(profile_points, radius):
    from fourpoint import get_roundness_profile
    phases = [
        0, 0, 4.12809636e+00, 4.76722652e+00, 4.74985898e-01, 4.96882293e+00, 2.50166547e+00, 4.95573223e+00,
        4.15941296e+00, 2.06000527e-01, 6.72622339e-01, 5.42008609e+00, 6.10669927e-01, 2.92664106e+00, 8.06480407e-02,
        5.06477712e+00, 4.29187534e+00, 3.15729467e+00, 2.01198012e+00, 8.78316033e-01, 4.61974764e-01, 1.26625800e+00,
        1.65608701e+00, 2.75569374e+00, 7.19873590e+00, 5.63227715e+00, 2.01876499e+00, 1.37086980e+00, 3.76452802e+00,
        4.63714463e+00, 1.50008254e+00, 6.67745487e+00, 8.35516439e-01, 2.45561686e+00, 4.01117909e+00, 1.97669359e+00,
        3.51151347e+00, 5.65000185e+00, 5.95336561e+00, 3.76918613e+00, 2.20214195e+00, 6.96449739e-01, 1.82764984e+00,
        3.41184820e+00, 2.49822614e+00, 1.44400829e+00, 3.65744742e+00, 1.10785096e+00, 1.47927148e+00, 2.09145913e+00,
        4.37208613e+00, 4.27888424e+00, 2.02481237e+00, 7.02855373e-01, 5.00768710e+00, 6.46682201e+00, 5.97777401e+00,
        6.10752444e+00, 3.08608594e+00, 3.86197304e+00, 4.11887277e+00, 4.59348700e+00, 2.12859775e+00, 1.34982399e+00,
        2.98169695e+00, 1.12948009e+00, 9.47181838e-01, 3.73934562e+00, 3.13230250e+00, 8.83752949e-01, 3.15862538e+00,
        1.73191616e+00, 4.02030606e+00, 6.18293076e+00, 4.15922231e+00, 6.28551963e+00, 3.15728358e+00, 1.78957478e+00,
        4.77041624e+00, 2.64131599e+00, 1.03636127e+00, 1.59382550e+00, 2.70848760e+00, 3.66300304e+00, 1.85418385e+00,
        6.19296779e-01, 5.96820705e+00, 5.41388044e+00, 2.55714851e+00, 3.98402948e+00, 1.32671603e+00, 3.56629218e+00,
        1.01068194e+00, 6.01515570e+00, 6.03172971e+00, 1.42415613e+00, 5.94646018e+00, 4.22817803e-03, 2.53156208e+00,
        6.39685605e+00, 1.54342525e+00
    ]

    fft = [0 + 0J] * (int(profile_points / 2))
    amplitude = -0.01
    fourier_adjusted_amplitude = amplitude / (2 / profile_points)

    fd = np.array([create_unit_complex(phase) for phase in phases]) * fourier_adjusted_amplitude

    for index, value in enumerate(fd):
        fft[index] = value

    fft[0] = 0 + 0J
    fft[1] = 0 + 0J
    reverse_offset_fft = np.conjugate(np.flip(fft[1:]))

    fft = np.append(np.append(fft, [0]), reverse_offset_fft)

    # fft = np.append(fft, np.conj((fft[1:])[::-1]))

    roundness, theta, r = get_roundness_profile(fft, profile_points)
    imags = sorted([i.imag for i in np.fft.ifft(fft)])

    r = [i + radius for i in r]

    return theta, r


def generate_reference_profile(profile_points, radius):
    from fourpoint import get_roundness_profile

    phase_deg = [
        0, 0, 9.0, 200.0, 253.0, 20.0, 55.0, 2.0, 137.0, 1.0, 161.0, 212.0, 24.0, 200.0, 245.0, 188.0, 16.0, 11.0,
        75.0, 211.0, 294.0, 165.0, 81.0, 104.0, 59.0, 91.0, 43.65, 157.0, 91.0, 321.0, 161.0
    ]

    fft = [0 + 0J] * (int(profile_points / 2))
    amplitude = -0.01
    fourier_adjusted_amplitude = amplitude / (2 / profile_points)

    fd = np.array([create_unit_complex(np.deg2rad(phase)) for phase in phase_deg]) * fourier_adjusted_amplitude

    for index, value in enumerate(fd):
        fft[index] = value

    fft[0] = 0 + 0J
    fft[1] = 0 + 0J
    reverse_offset_fft = np.conjugate(np.flip(fft[1:]))

    fft = np.append(np.append(fft, [0]), reverse_offset_fft)

    # fft = np.append(fft, np.conj((fft[1:])[::-1]))

    roundness, theta, r = get_roundness_profile(fft, profile_points)
    imags = sorted([i.imag for i in np.fft.ifft(fft)])

    # for i in imags:
    #     print(i)

    r = [i + radius for i in r]

    return theta, r


def generate_profile_signals(samples=1024,
                             cpm_points=None,
                             vertical_error=None,
                             horizontal_error=None,
                             angles_deg_errors=None):
    radius = 250
    polygon_points = ACCURACY
    theta, r = generate_reference_profile_100(polygon_points, radius)
    # theta, r = generate_reference_profile(polygon_points, radius)
    # theta, r = generate_profile(polygon_points, radius)

    if not cpm_points:
        t = np.linspace(0, 2 * np.pi, samples)
        cpm_points = polar_to_cartesian(t, [0.00] * samples)

    points = polar_to_cartesian(theta, r)
    polygon = Polygon(points)
    angles = [0, 38, 67, 180]
    # angles = [0, 37.06875, 65.88984375, 179.41640625]
    signals = generate_signals(
        polygon, angles,
        samples=samples,
        cpm_points=cpm_points,
        vertical_error=vertical_error,
        horizontal_error=horizontal_error,
        angles_deg_errors=angles_deg_errors)

    profile = [radius, list(theta)[::-1], r]
    # profile = [radius, list(theta), r]

    return signals, profile


def plot_profile(profile):
    theta, r = profile[1], profile[2]
    points = polar_to_cartesian(theta, r)
    polygon = Polygon(points)
    plot_polygon(polygon)


def plot_signals(signals):
    for i in signals:
        plt.plot(i)
    # plt.plot(signals[0])
    # plt.plot(signals[1])
    # plt.plot(signals[2])
    # plt.plot(signals[3])
    plt.show()


def plot_diameter(signals):
    ssignal = (signals[0] + signals[3])
    plt.plot(signals[0])
    plt.plot(signals[3])
    plt.plot(ssignal)
    plt.show()


def generate_circle_signals(ecc=0, samples=1024):

    points = 10240
    theta = np.linspace(0, 2 * np.pi, points)

    radius = 50
    r = [radius] * points

    points = polar_to_cartesian(theta, r)
    # print(points)
    # plt.plot(points)
    # plt.show()
    polygon = Polygon(points)

    signals = generate_signals(polygon, [0], samples=samples, origin=Point(0, ecc))

    profile = [radius, list(theta)[::-1], r]

    return signals, profile


def ecc_trig():

    import matplotlib.ticker as tck
    from mpl_toolkits.mplot3d import Axes3D
    eccs = np.linspace(0, 0.1, 10)
    X = []
    Y = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for ecc in eccs:
        signals, profile = generate_circle_signals(ecc=ecc)
        from fourpoint import average
        signals = [np.array(s) for s in signals]
        signals = np.array(average(signals))

        angle = np.linspace(0, 2 * np.pi, len(signals[0]))
        diff = signals[0] - (ecc * np.sin(np.linspace(0, 2 * np.pi, len(signals[0]))))

        # ax.plot(diff, angle, ecc, color="black")
        ax.plot(angle, diff, ecc, zdir="y", color="black")

        # Y.append(diff)
        # X.append(np.linspace(0, 2 * np.pi, len(signals[0])))

    # ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    # ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$e/d$')
    ax.set_zlabel(r'$S(\theta) - cos(\theta)$')
    # ax.set_zlabel('');

    plt.show()


def ecc_surf():

    import matplotlib.ticker as tck
    from mpl_toolkits.mplot3d import Axes3D
    eccs = np.linspace(0, 1, 20)
    Y = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for ecc in eccs:
        signals, profile = generate_circle_signals(ecc=ecc)
        from fourpoint import average
        signals = [np.array(s) for s in signals]
        signals = np.array(average(signals))

        angle = np.linspace(0, 2 * np.pi, len(signals[0]))
        diff = signals[0] - (ecc * np.sin(np.linspace(0, 2 * np.pi, len(signals[0]))))

        Y.append(diff)

    Y = np.array(Y)

    x = angle

    for index, ecc in enumerate(eccs):
        z = ecc
        x = angle
        # y = (Y[:, index])
        y = Y
        # x, y = zip(*sorted(zip(x, y)))
        ax.plot_trisurf(x, y, z)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$e/d$')
    ax.set_zlabel(r'$S(\theta) - cos(\theta)$')

    plt.show()


@timing
def test_timing():

    signals, profile = generate_profile_signals(ecc=ecc, samples=1024)
    from fourpoint import average
    signals = [np.array(s) for s in signals]
    signals = np.array(average(signals))
    print("generated")


if __name__ == "__main__":
    # ecc_trig()
    # ecc_surf()

    # signals, profile = generate_profile_signals(
    #     1024, cpm_points=None, vertical_error=None, horizontal_error=None, angles_deg_errors=None)

    # plt.plot(profile[2])
    # plt.plot(profile[2][::-1])
    # print(profile[2][0] - profile[2][-1])
    # print(profile[2][0] - profile[2][1])
    # plt.show()
    # plot_profile(profile)

    generate_reference_profile_100(1000, 200)
