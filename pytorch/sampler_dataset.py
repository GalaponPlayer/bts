import numpy as np
import argparse
import sys
import os
from tqdm import tqdm
from PIL import Image

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

def bilinear_sampler(img, depth, focal=707.0, base=0.54, ifBTS=True):
    if ifBTS:
        depth = np.array(depth) / 256.0
    else:
        depth = np.array(depth)
    
    input_h, input_w = depth.shape

    disp = base * focal / depth
    x_offset = disp.astype(np.int)

    left = np.array(img.resize((input_w, input_h)))
    left = left.transpose(2,0,1)

    w_idx = np.arange(0, input_w)
    idx = np.repeat(w_idx[None, :], input_h, axis=0)

    x0 = idx + x_offset
    x0 = np.clip(x0, 0, input_w - 1)

    x = idx + disp
    x = np.clip(x, 0.0, input_w - 1.0)

    x1 = x0 + 1
    x1 = np.clip(x1, 0, input_w - 1)

    x0 = np.repeat(x0[None, :], 3, axis=0)
    x1 = np.repeat(x1[None, :], 3, axis=0)

    pix_l = np.take_along_axis(left, x0, 2)
    pix_r = np.take_along_axis(left, x1, 2)

    x0 = np.clip(x0, 0, input_w - 2)
    dist_l = x - x0
    dist_r = x1 - x

    output = dist_r * pix_l + dist_l * pix_r

    output = output.transpose(1,2,0)
    right_img = Image.fromarray(np.uint8(output))

    return right_img

parser = argparse.ArgumentParser(description='bilinear sampler numpy implementation', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--data_dir')
parser.add_argument('--depth_dir', help='path to dir contain depth maps from BTS.')
parser.add_argument('--output_dir', help='path to dir wanna save images.')


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

def sample_dataset(args):
    left_path = os.path.join(args.data_dir, 'image_2')
    calib_path = os.path.join(args.data_dir, 'calib')
    depth_path = args.depth_dir

    left_names = os.listdir(left_path)
    calib_names = os.listdir(calib_path)
    depth_names = os.listdir(depth_path)
    left_names.sort()
    calib_names.sort()
    depth_names.sort()

    left_paths = [os.path.join(left_path, left_name) for left_name in left_names]
    calib_paths = [os.path.join(calib_path, calib_name) for calib_name in calib_names]
    depth_paths = [os.path.join(depth_path, depth_name) for depth_name in depth_names]

    num_images = len(left_paths)

    for left_name, calib_name, depth_name in tqdm(zip(left_paths, calib_paths, depth_paths), total=num_images):
        left = Image.open(left_name)
        calib = read_calib_file(calib_name)
        depth = Image.open(depth_name)

        input_width, input_height = left.size

        focal = float(calib['P2'][0])

        right = bilinear_sampler(left, depth, focal=focal)
        right = right.resize((input_width, input_height))
        right_name = os.path.join(args.output_dir, os.path.basename(left_name))
        right.save(right_name)


if __name__ == "__main__":
    sample_dataset(args)