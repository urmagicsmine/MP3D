import cv2
import os
import os.path as osp
import numpy as np
SLICE_INTERVAL = 2.5
import pdb

def load_multislice_gray_png_3DCE(imname, num_slice=9, slice_intv=5, window=None,
        zflip=False, target_intv=None):
    # Load single channel image for lesion detection
    def get_slice_name(img_path, delta=0):
        if delta == 0:
            return img_path
        delta = int(delta)
        name_slice_list = img_path.split(os.sep)
        slice_idx = int(name_slice_list[-1][:-4])
        img_name = '%03d.png' % (slice_idx + delta)
        full_path = os.path.join('./', *name_slice_list[:-1], img_name)

        # if the slice is not in the dataset, use its neighboring slice
        while not os.path.exists(full_path):
            # print('file not found:', img_name)
            delta -= np.sign(delta)
            img_name = '%03d.png' % (slice_idx + delta)
            full_path = os.path.join('./', *name_slice_list[:-1], img_name)
            if delta == 0:
                break
        return full_path

    def _load_data(img_name, delta=0):
        img_name = get_slice_name(img_name, delta)
        if img_name not in data_cache.keys():
            data_cache[img_name] = cv2.imread(img_name, -1)  #-1代表8位深度，原通道；0代表8位深度，1通道（灰度图）
            if data_cache[img_name] is None:
                print('file reading error:', img_name, os.path.exists(img_name))
                assert not data_cache[img_name] == None
        return data_cache[img_name]

    def _load_multi_data(im_cur, imname, num_slice, slice_intv, zfilp, target_intv):
        ims = [im_cur]
        # find neighboring slices of im_cur
        rel_pos = float(target_intv) / slice_intv
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        sequence_flag = True
        if zflip:
            if np.random.rand() > 0.5:
                sequence_flag = False
        if sequence_flag:  #TODO: This is hard to read. Needs fix
            if a == 0:  # required SLICE_INTV is a divisible to the actual slice_intv, don't need interpolation
                for p in range((num_slice - 1) // 2):
                    im_prev = _load_data(imname, - rel_pos * (p + 1))
                    im_next = _load_data(imname, rel_pos * (p + 1))
                    ims = [im_prev] + ims + [im_next]
                # when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
                if num_slice % 2 == 0:
                    im_next = _load_data(imname, rel_pos * (p + 2))
                    ims += [im_next]
            else:
                for p in range((num_slice - 1) // 2):
                    intv1 = rel_pos * (p + 1)
                    slice1 = _load_data(imname, - np.ceil(intv1))
                    slice2 = _load_data(imname, - np.floor(intv1))
                    im_prev = a * slice1 + b * slice2  # linear interpolation
    
                    slice1 = _load_data(imname, np.ceil(intv1))
                    slice2 = _load_data(imname, np.floor(intv1))
                    im_next = a * slice1 + b * slice2
                    ims = [im_prev] + ims + [im_next]
                # when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
                if num_slice % 2 == 0:
                    intv1 = rel_pos * (p + 2)
                    slice1 = _load_data(imname, np.ceil(intv1))
                    slice2 = _load_data(imname, np.floor(intv1))
                    im_next = a * slice1 + b * slice2
                    ims += [im_next]
        else:
            if a == 0:  # required SLICE_INTV is a divisible to the actual slice_intv, don't need interpolation
                for p in range((num_slice - 1) // 2):
                    im_next = _load_data(imname, - rel_pos * (p + 1))
                    im_prev = _load_data(imname, rel_pos * (p + 1))
                    ims = [im_prev] + ims + [im_next]
            else:
                for p in range((num_slice - 1) // 2):
                    intv1 = rel_pos * (p + 1)
                    slice1 = _load_data(imname, - np.ceil(intv1))
                    slice2 = _load_data(imname, - np.floor(intv1))
                    im_next = a * slice1 + b * slice2  # linear interpolation

                    slice1 = _load_data(imname, np.ceil(intv1))
                    slice2 = _load_data(imname, np.floor(intv1))
                    im_prev = a * slice1 + b * slice2
                    ims = [im_prev] + ims + [im_next]

        return ims
    # init target slice interval
    if target_intv is None:
        target_intv = SLICE_INTERVAL
    data_cache = {}
    im_cur = cv2.imread(imname, -1)
    num_slice = num_slice
    ims = _load_multi_data(im_cur, imname, num_slice, slice_intv, zflip, target_intv)
    ims = [im.astype(float) for im in ims]
    im = cv2.merge(ims)
    if window:
        im = im.astype(np.float32, copy=False) - 32768
        im = windowing(im, window)
    #im = windowing(im, [-1024, 1050])
    return im

def windowing(im, win):
    # Scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

