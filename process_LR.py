import os
import numpy as np
from PIL import Image

def process_LR(source_path, target_path):
    img_list = os.listdir(source_path)
    num_file, cnt = len(img_list), 0
    for img_filename in img_list:
        cnt += 1
        print "processing", str(cnt), "image file out of", num_file
        with Image.open(source_path + img_filename) as i:
            imgSize, rawData = i.size, i.tobytes()
            im = Image.frombytes('RGB', imgSize, rawData)
            im = im.resize((256,256), Image.ANTIALIAS)
            im.save(target_path + img_filename)

if __name__ == "__main__":
        process_LR("/cvgl2/u/hirose/dataset_depth/img_L_1/", "/cvgl2/u/hhlics/dataset_depth/img_L_1/")
        process_LR("/cvgl2/u/hirose/dataset_depth/img_R_1/", "/cvgl2/u/hhlics/dataset_depth/img_R_1/")