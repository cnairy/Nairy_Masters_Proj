#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:32:20 2022

@author: christian.nairy
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units

#plot balloon sounding
path = '/home/christian.nairy/capeex19/Balloon/20190803_1424/'

col_names = ['height','pressure', 'temperature','RH', 'speed', 'direction']

df = pd.read_fwf(path + 'B215B6-trimmed.txt',
                 skiprows=2, usecols=[4, 5, 6, 7, 8, 9], names=col_names)

df = df.replace('//', np.nan)
df = df.apply(pd.to_numeric, errors='coerce')


df = df.dropna(subset=('height','pressure','temperature','RH', 'speed', 'direction'), how='all'
               ).reset_index(drop=True)

# df['temperature'] = df.temperature[1175:] * -1

p = df['pressure'].values
T = df['temperature'].values
rh = df['RH'].values
wind_speed = df['speed'].values * units.knots
wind_dir = df['direction'].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)

# T[1175:] *= -1

T[1150:] *= -1

#Calc dew point
Td = T - ((100 - rh)/5.)

fig = plt.figure(figsize=(9, 9))
# add_metpy_logo(fig, 115, 100)
skew = SkewT(fig, rotation=45)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot.
skew.plot(p, T, 'r')
skew.plot(p, Td, 'b')
skew.plot_barbs(p[::100], u[::100], v[::100])
skew.ax.set_ylim(1000, 50)
skew.ax.set_xlim(-40, 40)
skew.ax.set_xlabel('Temperature (C)', fontsize = 12)
skew.ax.set_ylabel('Pressure (hPa)', fontsize = 12)

# Calculate LCL height and plot as black dot. Because `p`'s first value is
# ~1000 mb and its last value is ~250 mb, the `0` index is selected for
# `p`, `T`, and `Td` to lift the parcel from the surface. If `p` was inverted,
# i.e. start from low value, 250 mb, to a high value, 1000 mb, the `-1` index
# should be selected.
# lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
# skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

# Calculate full parcel profile and add to plot as black line
# prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
# skew.plot(p, prof, 'k', linewidth=2)

# Shade areas of CAPE and CIN
# skew.shade_cin(p, T, prof, Td)
# skew.shade_cape(p, T, prof)

# An example of a slanted line at constant T -- in this case the 0
# isotherm
skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Show the plot
plt.show()