#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:xvhuangjian
# time:2022/10/7

from mayavi import mlab
import numpy as np


def viz_mayavi(points, vals="distance"):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mlab.points3d(x, y, z,
                  z,  # Values used for Color
                  mode="point",
                  colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                  # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                  figure=fig,
                  )
    mlab.show()


if __name__ == '__main__':
    points = np.fromfile('/home/xhj/OpenPCDet/data/kitti/training/velodyne/000004.bin', dtype=np.float32).reshape([-1, 4]) #test and val
    viz_mayavi(points)
