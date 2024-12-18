import json
from glob import glob
from tqdm import tqdm
import os
import random


# data path 

# # resnet 
# data_dir = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/same_images_captions_0727'
# save_path = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/group_resnet_2nd_stage.json'

# UNI
# data_dir = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/same_images_captions_pathencoder_0727'
data_dir = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/same_images_aingle-captions_uni_0801'
save_path = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/group_uni_2nd_stage-0801.json'

json_file_paths = glob(os.path.join(data_dir, '*.json'))
print(f'**There are {len(json_file_paths)} files in total.')
random.shuffle(json_file_paths)

final_dict = {}
for path in tqdm(json_file_paths):

    # unique caption in one video with the fisrt set id
    captions_in_video = {}

    with open(path) as f:
        data = json.load(f)
    save_data = data.copy()

    for key, value in data.items():

        if key in save_data.keys():

            similar_set = []
            # assert caption has occur before
            for caption in value["captions"].keys():
                try:
                    similar_set_id = captions_in_video[caption]
                    similar_set.append(similar_set_id)
                except KeyError:
                    a = 0
            # similar sets with this set
            similar_set = list(set(similar_set))

            # every caption occurs for the first time
            if len(similar_set) == 0:
                for caption in value["captions"].keys():
                    captions_in_video.update({caption:key})

            else:
                save_set_id = similar_set[0]

                # group all similar sets in the first set

                # target set
                save_data[save_set_id]["images"].extend(save_data[key]["images"])
                save_data[save_set_id]["captions"].update(save_data[key]["captions"])
                for cap in save_data[key]["captions"]:
                    captions_in_video.update({cap:save_set_id})

                # after updating, delete the original set
                del save_data[key]

                # other similar sets
                for set_id in similar_set[1:]:
                    save_data[save_set_id]["images"].extend(save_data[set_id]["images"])
                    save_data[save_set_id]["captions"].update(save_data[set_id]["captions"])

                    for cap in save_data[set_id]["captions"]:
                        captions_in_video.update({cap:save_set_id})
                    del save_data[set_id]

    final_dict.update(save_data)
    # break
    
print(f'***Finally, we get {len(final_dict)} data groups for training.')
with open(save_path, 'w') as f0:
    json.dump(final_dict, f0, indent=2)