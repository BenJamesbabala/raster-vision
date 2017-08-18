import json
import argparse
from os import makedirs
from os.path import join, splitext

import numpy as np
import matplotlib as mpl
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
import rasterio
from scipy.misc import imsave


def get_boxes_from_geojson(json_path, image_dataset):
    with open(json_path, 'r') as json_file:
        geojson = json.load(json_file)
    features = geojson['features']
    boxes = []
    for feature in features:
        polygon = feature['geometry']['coordinates'][0]
        # Convert to pixel coords.
        polygon = np.array([image_dataset.index(p[0], p[1]) for p in polygon])

        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)

        boxes.append((xmin, ymin, xmax, ymax))

    # Remove duplicates. Needed for ships dataset.
    boxes = list(set(boxes))

    return boxes


def print_box_stats(boxes):
    print('# boxes: {}'.format(len(boxes)))
    np_boxes = np.array(boxes)

    width = np_boxes[:, 2] - np_boxes[:, 0]
    print('width (mean, min, max): ({}, {}, {})'.format(
          np.mean(width), np.min(width), np.max(width)))

    height = np_boxes[:, 3] - np_boxes[:, 1]
    print('height (mean, min, max): ({}, {}, {})'.format(
          np.mean(height), np.min(height), np.max(height)))


def make_debug_plot(output_debug_dir, window_box, box_ind, im):
    # draw rectangle representing box
    window_xmin, window_ymin, window_xmax, window_ymax = window_box
    debug_im = np.copy(im)

    debug_im[window_xmin, window_ymin:window_ymax, :] = 0
    debug_im[window_xmax - 1, window_ymin:window_ymax, :] = 0
    debug_im[window_xmin:window_xmax, window_ymin, :] = 0
    debug_im[window_xmin:window_xmax, window_ymax - 1, :] = 0

    debug_window_path = join(
        output_debug_dir, '{}.jpg'.format(box_ind))
    imsave(debug_window_path, debug_im)


def make_chips(image_path, json_path, output_dir, debug=False,
               chip_size=300):
    makedirs(output_dir, exist_ok=True)
    output_image_dir = join(output_dir, 'images')
    makedirs(output_image_dir, exist_ok=True)
    if debug is not None:
        output_debug_dir = join(output_dir, 'debug')
        makedirs(output_debug_dir, exist_ok=True)

    image_dataset = rasterio.open(image_path)
    boxes = get_boxes_from_geojson(json_path, image_dataset)
    print_box_stats(boxes)

    # TODO make boxes a set
    # iterate through them, but remove any boxes that are contained completely
    # in the window so we don't generate near-duplicate windows.
    for box_ind, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box

        # pick random window around box
        width = xmax - xmin
        rand_x = int(np.random.uniform(xmin - (chip_size - width), xmin))

        height = ymax - ymin
        rand_y = int(np.random.uniform(ymin - (chip_size - height), ymin))

        # grab window from tiff
        window = ((rand_x, rand_x + chip_size), (rand_y, rand_y + chip_size))
        im = np.transpose(
            image_dataset.read(window=window), axes=[1, 2, 0])
        # bgr-ir
        im = im[:, :, [2, 1, 0]]

        # save window
        window_path = join(output_image_dir, '{}.jpg'.format(box_ind))
        imsave(window_path, im)

        # transform box coordinates so they are in window frame of reference
        window_box = (xmin - rand_x, ymin - rand_y,
                      xmax - rand_x, ymax - rand_y)

        # TODO write window_box and filename to CSV file

        if debug:
            make_debug_plot(output_debug_dir, window_box, box_ind, im)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiff-path')
    parser.add_argument('--json-path')
    parser.add_argument('--output-dir')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--chip-size', type=int, default=300)
    args = parser.parse_args()

    print('tiff_path: {}'.format(args.tiff_path))
    print('json_path: {}'.format(args.json_path))
    print('output_dir: {}'.format(args.output_dir))
    print('debug: {}'.format(args.debug))
    print('chip_size: {}'.format(args.chip_size))

    return args


if __name__ == '__main__':
    args = parse_args()
    make_chips(
        args.tiff_path, args.json_path, args.output_dir, args.debug,
        args.chip_size)
