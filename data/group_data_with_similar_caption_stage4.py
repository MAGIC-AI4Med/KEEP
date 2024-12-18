import json
from glob import glob
from tqdm import tqdm
import os
import random
import copy
from fuzzywuzzy.utils import full_process

# data path 

# # resnet 
# data_dir = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/same_images_captions_0727'
# save_path = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/group_resnet_2nd_stage.json'

# UNI
# data_dir = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/same_images_captions_pathencoder_0727'
data_dir = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/group_uni_3nd_stage-diffrence'
save_path = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/group_uni_4nd_stage-diffrence.json'

json_file_paths = glob(os.path.join(data_dir, '*.json'))
print(f'**There are {len(json_file_paths)} files in total.')
# random.shuffle(json_file_paths)

final_dict = {}
for path in tqdm(json_file_paths):

    # unique caption in one video with the fisrt set id
    captions_in_video = {}

    with open(path) as f:
        data = json.load(f)
    # save_data = data.copy()
    # compare_data = data.copy()
    
    for key, value in data.items():

        captions = value['captions']
        save_captions = copy.deepcopy(captions)
        compare_captions = copy.deepcopy(captions)

        for caption_1 in captions.keys():
            if caption_1 in save_captions.keys():
                del compare_captions[caption_1]
                
                for caption_2 in captions.keys():
                    if caption_2 in compare_captions.keys():
                        token_cap1 = full_process(caption_1).split(' ')
                        token_cap2 = full_process(caption_2).split(' ')
                        # cap_sim = len(set(token_cap1).intersection(set(token_cap2)))/len(set(token_cap1 + token_cap2))
                        # if cap_sim > 0.9:

                        cap_sim = len(set(token_cap1).difference(set(token_cap2)))
                        if cap_sim == 2:
                            # print(key, key_1)
                            # save_data[key]["images"].extend(save_data[key_1]["images"])
                            # save_data[key]["captions"].update(save_data[key_1]["captions"])
                            del save_captions[caption_2]
                            del compare_captions[caption_2]
                            # break
        data.update({key:{'images':data[key]['images'], 'captions':save_captions}})
    final_dict.update(data)

print(f'***Finally, we get {len(final_dict)} data groups for training.')
with open(save_path, 'w') as f0:
    json.dump(final_dict, f0, indent=2)

            # captions_set_0 = value["captions"].keys()

            # del compare_data[key]

            # for caption_0 in captions_set_0:

                # for key_1, value_1 in data.items():
                    # if key_1 in compare_data.keys():
                        # if key_1 in save_data.keys():
                        # captions_set_1 = value_1["captions"].keys()

    #                     for caption_1 in captions_set_1:
    #                         token_cap1 = full_process(caption_0).split(' ')
    #                         token_cap2 = full_process(caption_1).split(' ')
    #                         cap_sim = len(set(token_cap1).intersection(set(token_cap2)))/len(set(token_cap1 + token_cap2))
    #                         if cap_sim > 0.9:
    #                             # print(key, key_1)
    #                             save_data[key]["images"].extend(save_data[key_1]["images"])
    #                             save_data[key]["captions"].update(save_data[key_1]["captions"])
    #                             del save_data[key_1]
    #                             del compare_data[key_1]
    #                             break

    # save_path = os.path.join(save_dir, os.path.basename(path))
    # with open(save_path, 'w') as f0:
    #     json.dump(save_data, f0, indent=2)  



        # if key in save_data.keys():

            # similar_set = []
            # # assert caption has occur before
            # for caption in value["captions"].keys():
            #     try:
            #         similar_set_id = captions_in_video[caption]
            #         similar_set.append(similar_set_id)
            #     except KeyError:
            #         a = 0
            # # similar sets with this set
            # similar_set = list(set(similar_set))

            # # every caption occurs for the first time
            # if len(similar_set) == 0:
        # for caption in value["captions"].keys():
            # captions_in_video.update({caption:key})


    # for item in captions_in_video.keys():
    #     for 

    #         else:
    #             save_set_id = similar_set[0]

    #             # group all similar sets in the first set

    #             # target set
    #             save_data[save_set_id]["images"].extend(save_data[key]["images"])
    #             save_data[save_set_id]["captions"].update(save_data[key]["captions"])
    #             for cap in save_data[key]["captions"]:
    #                 captions_in_video.update({cap:save_set_id})

    #             # after updating, delete the original set
    #             del save_data[key]

    #             # other similar sets
    #             for set_id in similar_set[1:]:
    #                 save_data[save_set_id]["images"].extend(save_data[set_id]["images"])
    #                 save_data[save_set_id]["captions"].update(save_data[set_id]["captions"])

    #                 for cap in save_data[set_id]["captions"]:
    #                     captions_in_video.update({cap:save_set_id})
    #                 del save_data[set_id]
    # save_path = os.path.join(save_dir, os.path.basename(path))
    # with open(save_path, 'w') as f0:
    #     json.dump(save_data, f0, indent=2)
    # final_dict.update(save_data)
    # break
    
# print(f'***Finally, we get {len(final_dict)} data groups for training.')
# with open(save_path, 'w') as f0:
#     json.dump(final_dict, f0, indent=2)