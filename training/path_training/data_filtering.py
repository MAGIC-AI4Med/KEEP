import os
import cv2
from PIL import Image
from glob import glob

data_dir = '/mnt/petrelfs/share_data/zhouxiao/pathology/datasets/retrieval/train/images'
# save_dir = '/mnt/petrelfs/share_data/zhouxiao/pathology/datasets/webpath/filter_images'

img_paths = glob(os.path.join(data_dir, '*'))

cnt = 0
for img_path in img_paths:
    # print(img_path)

    file_name = os.path.basename(img_path)
    # img_dir = os.path.join(data_dir, str(image_list[idx]))
        
        # if image_list[idx] not in preload_img_data:
    img = cv2.imread(img_path)
    if img is None:
        print(img_path)
    #     # print("Error loading image")
    else:

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # TODO loie: resize image -> 512* abc, large width==512
    # width, height = img.size
    # print(img.size)

        # if width > height:
        #     new_width = 512
        #     new_height = int((new_width / width) * height)
        # else:
        #     new_height = 512
        #     new_width = int((new_height / height) * width)

        # img_resized = img.resize((new_width, new_height), resample=0)


        # save_path = os.path.join(save_dir, file_name)
        # img_resized.save(save_path)
        # cnt += 1

    # break

# print(cnt)
        