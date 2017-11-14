import os, pickle
import numpy as np
from PIL import Image

def convert_depth_file(pickle_path, jpg_path):
# depth 480 rows x 640 cols
    pickle_list = os.listdir(pickle_path)
    num_file, cnt = len(pickle_list), 0
    for pickle_filename in pickle_list:
        cnt += 1
        print("processing", str(cnt), "pickle file out of", num_file)
        with open(pickle_path + pickle_filename, 'r') as f:
            np_arr = pickle.load(f)[50:,20:-40]
            np_arr[np_arr == 0] = 1e4
            np_arr = np_arr.astype(np.float32) / 1e4 * 255
            im = Image.fromarray(np.floor(np_arr))
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.resize((128,128), Image.ANTIALIAS)
            if im.mode != 'RGB':
                    im = im.convert('RGB')
            im.save(jpg_path + pickle_filename.replace(".pickle", ".jpg"))

if __name__ == "__main__":
        convert_depth_file("/cvgl2/u/hirose/dataset_depth/depth_1/", "/cvgl2/u/hhlics/dataset_depth/depth_1/")