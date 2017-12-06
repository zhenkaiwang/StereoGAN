from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
from PIL import Image
import fnmatch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="/cvgl/u/zackwang/StereoGAN/outputs/images", help="path to folder containing images")
a = parser.parse_args()
def main():
    mean_errors=[]
    for file_output in os.listdir(a.image_dir):
        if fnmatch.fnmatch(file_output, '*outputs.png'):
            file_target=file_output[:-11]+'targets.png'
            image_output=np.asarray(Image.open(a.image_dir+'/'+file_output))
            image_target=np.asarray(Image.open(a.image_dir+'/'+file_target))
            mean_errors.append(np.mean(np.abs(image_target-image_output)))
    mean_errors=np.array(mean_errors)
    print("Mean errors is "+ str(np.mean(mean_errors)) + " mm")

main()
