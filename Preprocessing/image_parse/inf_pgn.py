# File: inf_pgn_single_palette.py
from __future__ import print_function
import argparse
import os
import sys
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

# Make sure utils scripts are available
from utils.image_reade_inf import ImageReader
from utils.ops import *
from utils.model_pgn import PGNModel  # <--- THIS IS THE FIX
try:
    from utils.utils import label_colours, load
except ImportError:
    print("Error: Could not import 'label_colours' or 'load' from 'utils.utils'.")
    print("Please ensure the 'utils' directory and its contents are available.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="PGN Inference to generate a dual-use paletted PNG map.")
    parser.add_argument('-i', '--image', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('-o', '--output_path', type=str, default='./output_palette.png', help='Path to save the final paletted PNG file.')
    parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoint/CIHP_pgn', help='Path to the pre-trained model checkpoint.')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: Input image not found at '{args.image}'")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at '{args.checkpoint}'")
        sys.exit(1)

    image_list_inp = [args.image]
    N_CLASSES = 20
    RESTORE_FROM = args.checkpoint
    coord = tf.train.Coordinator()
    with tf.compat.v1.name_scope("create_inputs"):
        reader = ImageReader(image_list_inp, None, False, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
    image_batch = tf.stack([image, image_rev])
    h_orig, w_orig = tf.cast(tf.shape(image_batch)[1], dtype=tf.float32), tf.cast(tf.shape(image_batch)[2], dtype=tf.float32)
    scales = [0.50, 0.75, 1.0, 1.25, 1.50, 1.75]
    image_batch_scaled = [tf.image.resize(image_batch, [tf.cast(h_orig*s, dtype=tf.int32), tf.cast(w_orig*s, dtype=tf.int32)]) for s in scales]

    net = []
    with tf.compat.v1.variable_scope('', reuse=False):
        net.append(PGNModel({'data': image_batch_scaled[0]}, is_training=False, n_classes=N_CLASSES))
    with tf.compat.v1.variable_scope('', reuse=True):
        for img in image_batch_scaled[1:]:
            net.append(PGNModel({'data': img}, is_training=False, n_classes=N_CLASSES))

    parsing_out1 = [tf.image.resize(n.layers['parsing_fc'], tf.shape(image_batch)[1:3,]) for n in net]
    parsing_out2 = [tf.image.resize(n.layers['parsing_rf_fc'], tf.shape(image_batch)[1:3,]) for n in net]
    parsing_out1_all = tf.reduce_mean(tf.stack(parsing_out1), axis=0)
    parsing_out2_all = tf.reduce_mean(tf.stack(parsing_out2), axis=0)
    raw_output = tf.reduce_mean(tf.stack([parsing_out1_all, parsing_out2_all]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = tail_list[:14] + [tail_list[15], tail_list[14], tail_list[17], tail_list[16], tail_list[19], tail_list[18]]
    tail_output_rev = tf.reverse(tf.stack(tail_list_rev, axis=2), tf.stack([1]))
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    pred_all = tf.expand_dims(tf.argmax(tf.expand_dims(raw_output_all, dim=0), axis=3), dim=3)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        loader = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
        if load(loader, sess, RESTORE_FROM):
            print("✅ Model loaded successfully.")
        else:
            print("❌ Failed to load model."); sys.exit(1)

        threads = tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)
        print(f"🚀 Processing image: {args.image}")
        parsing_result, = sess.run([pred_all])

        print("🎨 Creating and saving paletted PNG...")
        raw_label_map = parsing_result[0, :, :, 0].astype(np.uint8)
        palette_image = Image.fromarray(raw_label_map, mode='P')
        flat_palette = [value for color in label_colours for value in color]
        palette_image.putpalette(flat_palette)
        palette_image.save(args.output_path)
        print(f"✅ Paletted image saved to: {args.output_path}")

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
