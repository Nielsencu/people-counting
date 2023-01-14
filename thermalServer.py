# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import struct
import numpy as np
import matplotlib.pyplot as plt

import cv2, socket, pickle
s=socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
ip="192.168.175.49"
port=7000
s.bind((ip,port))

# If no device is found, exit the script

plt.rcParams.update({'font.size':16})
fig_dims = (12,9) # figure size
fig,ax = plt.subplots(figsize=fig_dims) # start figure
pix_res = (8,8) # pixel resolution
zz = np.zeros(pix_res) # set array with zeros first
im1 = ax.imshow(zz,vmin=15,vmax=40) # plot image, with temperature bounds
cbar = fig.colorbar(im1,fraction=0.0475,pad=0.03) # colorbar
cbar.set_label('Temperature [C]',labelpad=10) # temp. label
fig.canvas.draw() # draw figure

ax_bgnd = fig.canvas.copy_from_bbox(ax.bbox) # background for speeding up runs
fig.show() # show figure
fireThreshold = 30

while True:
    x=s.recvfrom(1000000)
    clientip = x[1][0]
    pixels=x[0]
    print(pixels)
    pixels=struct.unpack('<64f', pixels)
    print(pixels)
    print(type(pixels))
    # pixels = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
    maxTemp = np.max(pixels)
    pixels = np.flip(pixels, axis=0)
    print("Max temperature is ", maxTemp, " fire detected? ", maxTemp > fireThreshold)
    # T_thermistor = amg.read_thermistor() # read thermistor temp
    fig.canvas.restore_region(ax_bgnd) # restore background (speeds up run)
    pixels = np.reshape(pixels, pix_res)
    im1.set_data(pixels) # update plot with new temps
    for row in pixels:
        # Pad to 1 decimal place
        print(["{0:.1f}".format(temp) for temp in row])
        print("")
    print("\n")
    ax.draw_artist(im1) # draw image again
    fig.canvas.blit(ax.bbox) # blitting - for speeding up run
    fig.canvas.flush_events() # for real-time plot
    # print("Thermistor Temperature: {0:2.2f}".format(T_thermistor)) # print thermistor temp