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

OFFSET = .05
# FILTERLEVEL = 110
# FILTERLEVEL = 30
FILTERLEVEL = 110

X_OFFSET = 0  # to control how far the scale is from the plot (axes coordinates)


def add_scale(ax):
    # add extra axes for the scale
    rect = ax.get_position()
    rect = (
        rect.xmin - X_OFFSET,
        rect.ymin + rect.height / 2,  # x, y
        rect.width,
        rect.height / 2)  # width, height
    scale_ax = ax.figure.add_axes(rect)
    # hide most elements of the new axes
    for loc in ['right', 'top', 'bottom']:
        scale_ax.spines[loc].set_visible(False)
    scale_ax.tick_params(bottom=False, labelbottom=False)
    scale_ax.patch.set_visible(False)  # hide white background

    # adjust the scale
    scale_ax.spines['left'].set_bounds(ax.get_yticks()[0], ax.get_yticks()[1])
    scale_ax.set_yticks([ax.get_yticks()[0], ax.get_yticks()[1]])
    scale_ax.set_ylim(ax.get_rorigin(), ax.get_yticks()[1])


def polar_plot(data, label=None, offset=None, plot_circles=False, title=None):
    def round_nearest(x, a):
        return round(x / a) * a

    # r_max = round_nearest(data[0], 0.01)
    # print(r_max)
    r_max = 0.05

    roundness, theta, r = data
    # offset = 2 * abs(min(r))

    r = [i + abs(min(r)) for i in r]


    r = list(r)
    theta = list(theta)
    r.append(r[0])
    theta.append(theta[0])

    ax = plt.subplot(111, projection='polar')
    line, = ax.plot(theta, r, linewidth=1.5, label=label)

    if offset:
        ax.set_rorigin(-1 * offset)
        # ax.legend(loc=10, prop={"size": 8})
        ax.legend(loc=10, prop={"size": 18})
    # add_scale(ax)

    if plot_circles:
        ax.plot(theta, [min(r)] * len(theta), linewidth=2, color='r', ls='dashed')

    step = 0.01
    ax.set_rticks(np.arange(0, round_nearest((r_max + step), step), step))

    # ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.set_rlabel_position(62.5)  # get radial labels away from plotted line
    ax.grid(True)

    if not title:
        # ax.set_title("Roundness profile", va='bottom')
        pass
    else:
        ax.set_title("Roundness profile {}".format(title), va='bottom')
    plt.show()

def barchart(fft, filterlevel=FILTERLEVEL):
    scale = 2 / len(fft)

    x = np.arange(0, filterlevel + 1)
    y = fft[0:filterlevel + 1]

    f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

    ticks = x[1:filterlevel + 1]
    ticklabels = list(x[1:filterlevel + 1])
    ticklabels[0] = "e"
    ticklabels[1] = ""
    for i in range(2, len(ticks), 2):
        ticklabels[i] = ""

    xlim = 0

    display_ecc = True
    if not display_ecc:
        ticklabels[1] = "2"
        ticks = ticks[1:]
        ticklabels = ticklabels[1:]
        x = x[1:]
        y = y[1:]
        xlim = 1

    ax1.set_xticks(ticks)
    ax1.set_xticklabels(ticklabels)
    ax1.set_xlim(xlim, filterlevel + 1)

    ax1.bar(x, np.abs(y) * scale, align='center', alpha=0.8)
    ax1.set_title('Harmonic amplitudes and phases')
    ax1.set_ylabel('Amplitude')
    # ax1.set_ylim([0, 0.015])

    ax2.set_xlim(xlim, filterlevel + 1)

    ax2.bar(x, np.angle(y), align='center', alpha=0.8)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(ticklabels)

    ax2.set_xlabel('Harmonic')
    ax2.set_ylabel('Phase (rad)')
    ax2.set_yticks([-1 * np.pi, 0, np.pi])
    ax2.set_yticklabels([r"$-\pi$", "$0$", r"$\pi$"])
    ax2.set_ylim(-1.1 * np.pi, 1.1 * np.pi)

if __name__ == "__main__":
    barchart([0, 0,] + [0.1]*1000, 100)
    plt.show()
