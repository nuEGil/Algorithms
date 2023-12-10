import os
import cv2
import numpy as np
from pathlib import Path

class ScanMake():
    def __init__(self, hdir, stride = 1, win = 1):
        # defaults to scanning in the X direction
        self.hdir = hdir # where to save the output pictures
        self.oshape = (1140, 912) # so rows by columns not x,y
        self.roi = (108, 387, 205, 808) # top left corner row, column, width, height
        self.mu_per_pix = 26 # microns per pixel
        self.stride = stride # number of pixels to skip by in the scan
        self.win = win # window size on the scanner to make use of.

    def ScanSequence(self):
        # default behavior is going to be an x scan
        print('generating scan sequence')
        # Default behavior is to get an X scan
        self.BOX = []
        for ii in range(self.roi[1], self.roi[1] + self.roi[2] , self.stride):
            box = np.zeros(self.oshape, dtype = np.uint8) # initialize the holder
            box[self.roi[0] : self.roi[0] + self.roi[-1] , ii : ii + self.win] = 255
            self.BOX.append(box)

    def DumpImages(self):
        # save the images to a directory
        print('dumping')
        for bi, bb in enumerate(self.BOX):
            cv2.imwrite(os.path.join(self.hdir,'mask_{0:05d}.png'.format(bi)), bb) # write the image

class YScanMake(ScanMake):
    def __init__(self, hdir, stride = 1, win = 1):
        # defaults to scanning in the X direction
        self.hdir = hdir # where to save the output pictures
        self.oshape = (1140, 912) # so rows by columns not x,y
        self.roi = (108, 387, 205, 808) # top left corner row, column, width, height
        self.mu_per_pix = 26 # microns per pixel
        self.stride = stride # number of pixels to skip by in the scan
        self.win = win # window size on the scanner to make use of.

    def ScanSequence(self):
        # default behavior is going to be an x scan
        print('generating scan sequence')
        # Default behavior is to get an X scan
        self.BOX = []
        for ii in range(self.roi[0], self.roi[0] + self.roi[3] , self.stride):
            box = np.zeros(self.oshape, dtype = np.uint8) # initialize the holder
            box[ ii : ii + self.win, self.roi[1] : self.roi[1] + self.roi[2]] = 255
            self.BOX.append(box)

def worker():
    print('line 0')
    # set output directory and check if it exists
    hdir_ = 'ScanningPatterns/XScan_stride-1_win-3/'
    if not os.path.exists(hdir_):
        os.makedirs(hdir_) # make the directory if it doesnt exist
    CHULO = ScanMake(hdir_, stride = 1, win = 3)
    CHULO.ScanSequence()
    CHULO.DumpImages()

def Yworker():
    print('line 0')
    # set output directory and check if it exists
    hdir_ = 'ScanningPatterns/YScan_stride-1_win-3/'
    if not os.path.exists(hdir_):
        os.makedirs(hdir_) # make the directory if it doesnt exist
    CHULO = YScanMake(hdir_, stride = 1, win = 3)
    CHULO.ScanSequence()
    CHULO.DumpImages()

if __name__ == '__main__':
    print('working')
    Yworker()
