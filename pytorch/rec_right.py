# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
from torch.serialization import save
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from bts_dataloader import *

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from bts_dataloader import *

def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line)==0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--input_dir', type=str, help='file dir from which reconst right views')
parser.add_argument('--output_dir', type=str, help='output directory')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    #dataloader = BtsDataLoader(args, 'test')

    print(params.input_dir)

    image_path = os.path.join(params.input_dir, 'image_2')
    calib_path = os.path.join(params.input_dir, 'calib')

    img_names = os.listdir(image_path)
    calib_names = os.listdir(calib_path)
    img_names.sort()
    calib_names.sort()

    img_paths = [os.path.join(image_path, file_name) for file_name in img_names]
    calib_paths = [os.path.join(calib_path, calib_name) for calib_name in calib_names]

    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    #num_test_samples = get_num_lines(args.filenames_file)
    num_test_samples = len(img_names)

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []
    pred_8x8s = []
    pred_4x4s = []
    pred_2x2s = []
    pred_1x1s = []

    toTensor = ToTensor(params.mode)

    start_time = time.time()
    with torch.no_grad():
        for img_name, calib_name in tqdm(zip(img_paths, calib_paths)):
            if img_name == img_paths[100]:
                break
            image = np.asarray(Image.open(img_name), dtype=np.float32) / 255.0
            height = image.shape[0]
            width = image.shape[1]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

            calib = read_calib_file(calib_name)
            focal = float(calib['P2'][0])

            image = toTensor.to_tensor(image)
            focal = torch.tensor([focal])
            image = torch.reshape(image, (1, 3, 352, 1216))

            sample = {'image': image, 'focal': focal}

            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            # Predict
            # print('rec main')
            # print(image.shape)
            # print(focal.shape)
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
            pred_depths.append(depth_est.cpu().numpy().squeeze())
            pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
            pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
            pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
            pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    save_name = 'result_' + args.model_name
    save_name = params.output_dir
    
    print('Saving result pngs..')
    if not os.path.exists(os.path.dirname(save_name)):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    for s in tqdm(range(num_test_samples)):
        if s == 98:
            break
        if args.dataset == 'kitti':
            filename_pred_png = save_name + '/raw/' + img_names[s]
            filename_cmap_png = save_name + '/camp/' + img_names[s]
            filename_image_png = save_name + '/rgb/' + img_names[s]
        
        #rgb_path = os.path.join(args.data_path, './' + lines[s].split()[0])
        #image = cv2.imread(rgb_path)
        
        pred_depth = pred_depths[s]
        pred_8x8 = pred_8x8s[s]
        pred_4x4 = pred_4x4s[s]
        pred_2x2 = pred_2x2s[s]
        pred_1x1 = pred_1x1s[s]
        
        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
            pred_depth_scaled = pred_depth * 256.0
        else:
            pred_depth_scaled = pred_depth * 1000.0
        
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print('saved : ' + filename_pred_png)
        print(pred_depth_scaled.shape)
        
        if args.save_lpg:
            cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            if args.dataset == 'nyu':
                plt.imsave(filename_gt_png, np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
                pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
                plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')
                pred_8x8_cropped = pred_8x8[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8_cropped), cmap='Greys')
                pred_4x4_cropped = pred_4x4[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4_cropped), cmap='Greys')
                pred_2x2_cropped = pred_2x2[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2_cropped), cmap='Greys')
                pred_1x1_cropped = pred_1x1[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1_cropped), cmap='Greys')
            else:
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1), cmap='Greys')
    
    return


if __name__ == '__main__':
    test(args)
