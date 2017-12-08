from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import pickle
import json
import glob
import random
import collections
import math
import time
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="/cvgl2/u/hhlics/dataset_depth_full/", help="path to folder containing images")# done: set default path to the dataset
parser.add_argument("--depth_dir", default="/cvgl2/u/hhlics/dataset_depth_full/", help="path to folder containing images")# done: set default path to the dataset
parser.add_argument("--test_image_dir",default="/cvgl2/u/hhlics/dataset_depth_full", help="path to folder containing test images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", default= "outputs", help="where to put output files") # done: set default path for the output directory
parser.add_argument("--checkpoint", default="ckpt", help="directory with checkpoint to resume training from or use for testing") # done: set default path for the checkpoint directory
parser.add_argument("--seed", type=int)

parser.add_argument("--max_steps", type=int,help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=2000, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=2000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=64, help="number of images in batch") #set default batch size to be 16
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg","pickle"]) # done: add pickle type for depth.
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = a.scale_size

Examples = collections.namedtuple("Examples", "pathsL, pathsR, pathsD, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def preprocess(image):
    # done
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    # done
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def conv(batch_input, out_channels, stride): 
    # done
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    # done
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    # done
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        # input:[batch , height, width, channels]
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    # done
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


def check_image(image):
    # done
    #image size [number of image, height, weight, channels=2]
    assertion = tf.assert_equal(tf.shape(image)[-1], 2, message="image must have 2 channels, 1st channel must be left image, 2nd channel must be right image")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 2 so that you can unstack the channels
    shape = list(image.get_shape())
    shape[-1] = 2
    image.set_shape(shape)
    return image

def load_examples():
    # TODO: Rewrite this part
    # For inputs data: load all left and right image, convert into gray and resize into [number of image, 256, 256, 2], 
    # with the first channel be left image , the second channnel be right image. Normalize into [0,1), using tf.image.convert_image_dtype.
    # Then preprocess it. Just as what it is done in the code. You can also crop the fish eye image if necessary.
    # For targets data: Load the the depth data and resize into [number of image, 256, 256, 1]. 
    # Original depth data is pickle format, with each value in millimeter and dtype=uint16. 
    # Need to be convert to float32 and normalized into [0,1). The maximum detection range is 10000mm. So normalize the depth data with this value.
    # Then preprocess it. Just as what it is done in the code.
    if a.mode == 'test':
        if a.test_image_dir is None or not os.path.exists(a.test_image_dir):
            raise Exception("test_image_dir does not exist")

        input_L_paths = glob.glob(os.path.join(a.test_image_dir + "/img_L_test/", "*.jpg"))
        input_R_paths = glob.glob(os.path.join(a.test_image_dir + "/img_R_test/", "*.jpg"))
        depth_paths = glob.glob(os.path.join(a.test_image_dir + "/depth_test/", "*.jpg"))
        decode = tf.image.decode_jpeg

        if len(input_L_paths) == 0:
            raise Exception(a.test_image_dir + "/img_L_test/" + " contains no image files")
        if len(input_R_paths) == 0:
            raise Exception(a.test_image_dir + "/img_R_test/" + " contains no image files")
        if len(depth_paths) == 0:
            raise Exception(a.test_image_dir + "/depth_test/" + " contains no depth files")
    else:
        if a.input_dir is None or not os.path.exists(a.input_dir):
            raise Exception("input_dir does not exist")

        input_L_paths = glob.glob(os.path.join(a.input_dir + "/img_L_train/", "*.jpg"))
        input_R_paths = glob.glob(os.path.join(a.input_dir + "/img_R_train/", "*.jpg"))
        depth_paths = glob.glob(os.path.join(a.depth_dir + "/depth_train/", "*.jpg"))
        decode = tf.image.decode_jpeg

        if len(input_L_paths) == 0:
            raise Exception(a.input_dir + "/img_L_train/" + " contains no image files")
        if len(input_R_paths) == 0:
            raise Exception(a.input_dir + "/img_R_train/" + " contains no image files")
        if len(depth_paths) == 0:
            raise Exception(a.depth_dir + "/depth_train/" + " contains no depth files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_L_paths):
        input_L_paths = sorted(input_L_paths, key=lambda path: int(get_name(path)))
    if all(get_name(path).isdigit() for path in input_R_paths):
        input_R_paths = sorted(input_R_paths, key=lambda path: int(get_name(path)))
    if all(get_name(path).isdigit() for path in depth_paths):
        depth_paths = sorted(depth_paths, key=lambda path: int(get_name(path)))
    else:
        input_L_paths = sorted(input_L_paths)
        input_R_paths = sorted(input_R_paths)
        depth_paths = sorted(depth_paths)
    #for i in range(len(input_L_paths)):
    #    print('path '+ str(i) + ': ' + input_L_paths[i]+'\n'+input_R_paths[i]+'\n'+depth_paths[i]+'\n')

    with tf.name_scope("load_images"):
        path_L_queue = tf.train.string_input_producer(input_L_paths, shuffle=True, seed=42)#a.mode == "train")
        path_R_queue = tf.train.string_input_producer(input_R_paths, shuffle=True, seed=42)#a.mode == "train")
        depth_queue = tf.train.string_input_producer(depth_paths, shuffle=True, seed=42)#a.mode == "train")
        reader = tf.WholeFileReader()
        paths_L, contents_L = reader.read(path_L_queue)
        paths_R, contents_R = reader.read(path_R_queue)
        paths_depth, contents_depth = reader.read(depth_queue)
        paths_L.set_shape([])
        paths_R.set_shape([]) 
        paths_depth.set_shape([]) 
		# offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
		# r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        raw_input_L = decode(contents_L,channels=1)
        raw_input_R = decode(contents_R,channels=1)
        raw_input_depth = decode(contents_depth,channels=1)
        print(raw_input_L.get_shape().as_list())
        print(raw_input_depth.get_shape().as_list())
        raw_input_L = tf.image.crop_to_bounding_box(raw_input_L, 90, 90, 300, 300)
        raw_input_R = tf.image.crop_to_bounding_box(raw_input_R, 90, 90, 300, 300)
        raw_input_depth = tf.image.crop_to_bounding_box(raw_input_depth, 90, 90, 300, 300)
        raw_input_L = tf.image.resize_images(raw_input_L, size=(a.scale_size,a.scale_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        raw_input_R = tf.image.resize_images(raw_input_R, size=(a.scale_size,a.scale_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        raw_input_depth = tf.image.resize_images(raw_input_depth, size=(a.scale_size,a.scale_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        raw_input_L = tf.image.convert_image_dtype(raw_input_L, dtype=tf.float32)
        raw_input_R = tf.image.convert_image_dtype(raw_input_R, dtype=tf.float32)
        raw_input_depth = tf.image.convert_image_dtype(raw_input_depth, dtype=tf.float32)
        print(raw_input_L.get_shape().as_list())
        print(raw_input_depth.get_shape().as_list())
        if (len(raw_input_L.get_shape().as_list()) != 3):
            raise Exception("len(raw_input_L.get_shape().as_list()) != 3")
        
        raw_input_LR = tf.concat([raw_input_L, raw_input_R], 2)

        if (len(raw_input_LR.get_shape().as_list()) != 3):
            raise Exception("len(raw_input_L.get_shape().as_list()) != 3")
        
        assertion = tf.assert_equal(tf.shape(raw_input_LR)[-1], 2, message="image does not have 2 channels") # check the image data has 2 channels
        with tf.control_dependencies([assertion]):
            raw_input_LR = tf.identity(raw_input_LR)

        raw_input_LR.set_shape([a.scale_size, a.scale_size, 2]) #Set the image channels size to be 2
        raw_input_depth.set_shape([a.scale_size, a.scale_size, 1]) 

        a_images = preprocess(raw_input_LR)
        b_images = preprocess(raw_input_depth)

    input_images, target_images = [a_images, b_images]

    paths_L_batch, paths_R_batch, paths_depth_batch, inputs_batch, targets_batch = tf.train.batch([paths_L, paths_R, paths_depth, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_L_paths) / a.batch_size))

    return Examples(
        pathsL=paths_L_batch,
        pathsR=paths_R_batch,
        pathsD=paths_depth_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_L_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    # done
    layers = []

    # encoder_1: [batch, 256, 256, in_channels=2] => [batch, 128, 128, ngf]
    assertion = tf.assert_equal(tf.shape(generator_inputs)[-1], 2, message="image does not have 2 channels")
    with tf.control_dependencies([assertion]):
            generator_inputs = tf.identity(generator_inputs)
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    # done
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # inputs: [batch, height, width, 2] & targets:[batch, height, width, 1]=> [batch, height, width, 3]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, 3] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)
        print('generator output shape: ', outputs.get_shape().as_list())

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # [batch, height, width, 2+1] => [batch, 14, 14, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # [batch, height, width, 2+1] => [batch, 14, 14, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, step=None):
    # to do: 
    # Resize both the target and output data into original size and save as png file.
    if a.mode == "test":
        image_dir = os.path.join(a.output_dir, "test_results")
    else:
        image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["pathsL"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputsL","inputsR", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputsL", "inputsR","outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if tf.__version__.split('.')[0] != "1":
        raise Exception("Tensorflow version 1 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode=="export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"ngf", "ndf"}
        with open(os.path.join(a.output_dir, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # to do:
        # need to rewrite export mode, with the correct data format and size.

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert RGB to grayscale
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 3), lambda: tf.image.rgb_to_grayscale(input_image), lambda: input_image)

        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 1])
        input_image=tf.concat([input_image,input_image],axis=2)
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = deprocess(create_generator(preprocess(batch_input), generator_outputs_channels=1))# done: outputs channel should be 1

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.checkpoint, "export.meta"))
            export_saver.save(sess, os.path.join(a.checkpoint, "export"), write_meta_graph=True)

        return

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    def convert(image,size):
        # if a.aspect_ratio != 1.0:
            # # upscale to correct aspect ratio
            # size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
        # image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    print('input shape: ', inputs.get_shape().as_list())
    print('target shape: ', targets.get_shape().as_list())
    print('output shape: ', outputs.get_shape().as_list())
    with tf.name_scope("convert_inputs"):
        converted_inputs_L = convert(tf.expand_dims(inputs[:,:,:,0],3),size=(a.scale_size,a.scale_size))
        converted_inputs_R = convert(tf.expand_dims(inputs[:,:,:,1],3),size=(a.scale_size,a.scale_size))

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets,size=(a.scale_size,a.scale_size))

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs,size=(a.scale_size,a.scale_size))

    print('converted input shape: ', converted_inputs_L.get_shape().as_list())
    print('converted target shape: ', converted_targets.get_shape().as_list())
    print('converted output shape: ', converted_outputs.get_shape().as_list())

    with tf.name_scope("encode_images"):
        display_fetches = {
            "pathsL": examples.pathsL,
            "pathsR": examples.pathsR,
            "pathsD": examples.pathsD,
            "inputsL": tf.map_fn(tf.image.encode_png, converted_inputs_L, dtype=tf.string, name="inputL_pngs"),
            "inputsR": tf.map_fn(tf.image.encode_png, converted_inputs_R, dtype=tf.string, name="inputR_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
        # display_fetches = {
        #     "pathsL": examples.pathsL,
        #     "pathsR": examples.pathsR,
        #     "pathsD": examples.pathsD,
        #     "inputsL": converted_inputs_L,
        #     "inputsR": converted_inputs_R,
        #     "targets": converted_targets,
        #     "outputs": converted_outputs,
        # }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputsL", tf.image.convert_image_dtype(converted_inputs_L,dtype=tf.uint8))
        tf.summary.image("inputsR", tf.image.convert_image_dtype(converted_inputs_R,dtype=tf.uint8))

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", tf.image.convert_image_dtype(converted_targets,dtype=tf.uint8))

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", tf.image.convert_image_dtype(converted_outputs,dtype=tf.uint8))

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)
    tf.summary.merge_all()

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            print("loading model from checkpoint: ", checkpoint)
            if checkpoint is not None:
                saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)

            print("wrote index at", index_path)
        else:
            # training
            start = time.time()
            print("enter training mode, max_steps: ", max_steps)

            for step in range(max_steps):
                # print('step: ', step)
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss ", results["discrim_loss"])
                    print("gen_loss_GAN ", results["gen_loss_GAN"])
                    print("gen_loss_L1 ", results["gen_loss_L1"])
                    print("gen_loss ",results["gen_loss_GAN"] * a.gan_weight + results["gen_loss_L1"] * a.l1_weight)

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.checkpoint, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
