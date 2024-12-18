import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm
from glob import glob
import numpy as np
import torch.nn.functional as F
import pandas as pd 
import json
import timm
import random
import time

def model_load(model_name):

    if model_name == 'uni':
        local_dir = "/mnt/hwfile/medai/zhouxiao/model/model_zoo/UNI_model"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        model = model.to(device)
        model.eval()
        
    else: # resnet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.resnet50(pretrained=True)
        model = model.to(device)
        model.eval()
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        
    return model
 
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform()
    
    def __len__(self):
        return len(self.image_paths)
    
    def transform():
        transform = transforms.Compose(
            [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform
                
    def __getitem__(self, index):

        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, img_path

def cosine_simi(features):
    norms = features.norm(p=2, dim=1, keepdim=True)  
    normalized_features = features / norms  
    similarity_matrix = torch.mm(normalized_features, normalized_features.t())
    return similarity_matrix


# data path and load data
prefix = '/mnt/hwfile/medai/zhouxiao/data/pathology/training_datasets/crop_clean_images_512_modify/'
save_dir = '/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/same_images_captions_pathencoder_0727'

with open('/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/filtered_quilt_image_capions.json') as f:
    quilt_image_captions = json.load(f)

with open('/mnt/hwfile/medai/sunluoyi/pathology/code/process_data/__profile_quilt1m_cap_img_hierarchy_label.json') as fb:
    captions_labels = json.load(fb)

df = pd.read_csv('/mnt/hwfile/medai/zhouxiao/data/pathology/training_datasets/original_quilt_data.csv')
df = df[['video_id', 'image_path']]
video_ids = df['video_id'].tolist()
video_ids = list(set(video_ids))

# choose model 
model_thresholds = {'resnet':0.99, 'uni':0.95}
model_name = 'resnet' # or 'uni'
model = model_load(model_name)


transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# process data in same video
for video_id in tqdm(video_ids):

    video_images_captions = {}
    save_path = os.path.join(save_dir, f'{video_id}.json')
    captions_in_videos = []

    if not os.path.exists(save_path):

        # all images in one video
        filtered_df = df[df['video_id'] == video_id]
        image_list = filtered_df['image_path'].tolist()

        image_paths = []
        for item in image_list:
            image_path = prefix + item
            if os.path.exists(image_path):
                try:
                    # only process the images we reserved
                    caption = quilt_image_captions[item]
                    captions_in_videos.append(caption)
                    image_paths.append(image_path)
                except KeyError:
                    a = 0
        
        # extract image features
        if not len(image_paths) == 0:
            dataset = ImageDataset(image_paths, transform=transform)
            data_loader = DataLoader(dataset, batch_size=len(image_list), shuffle=False)

            with torch.no_grad():
                for inputs, paths in data_loader:
                    inputs = inputs.to(device)
                    features = model(inputs)
                    features = features.view(features.size(0), -1)
                    file_names = [os.path.basename(path) for path in paths]

            # calculate similarity and save item over threshold
            similarity = cosine_simi(features)
            index_list = list(range(len(file_names)))
            # positive_index[0] is the row index, and positive_index[0] is the column index
            positive_index = torch.where(similarity > model_thresholds[model_name])

            group_id = 0
            save_images_index = []
            for i in range(len(file_names)):

                set_images = []
                # set every row index as query
                # and load the column index list
                # positive_index_group[0] is the column index list
                positive_index_group = torch.where(positive_index[0]==i)

                # when all image processed, early close
                if len(save_images_index) == len(file_names):
                    break

                # determine whether the image has been processed in the previous process
                elif not i in save_images_index:

                    # just similar with itself 
                    if len(positive_index_group[0]) == 1:

                        set_id = f'{video_id}_{str(group_id).zfill(5)}'

                        # process starting, save image index in processed list(save_images_index)
                        # and add image_name in set
                        save_images_index.append(i)
                        set_images.append(file_names[i])
                        group_id += 1

                        try:
                            # read caption and its labels
                            # maybe the caption has been delete in the last process TEXT_FILTERING
                            caption = captions_in_videos[i]
                            caption_info = captions_labels[caption]
                            del caption_info["images"]
                            caption_dict = {"images":set_images, "captions":{caption:caption_info}}
                            video_images_captions.update({set_id:caption_dict})
                        except KeyError:
                            a = 1

                    # just similar with others 
                    else:
                        
                        # save query image information in set first
                        save_images_index.append(i)
                        set_images.append(file_names[i])
                        captions_in_set = []
                        captions_in_set.append(captions_in_videos[i])

                        set_id = f'{video_id}_{str(group_id).zfill(5)}'
                        group_id += 1

                        in_captions_labels = {}
                        for j in range(len(positive_index_group[0])):

                            j_in_video = positive_index[1][positive_index_group[0][j]]

                            # to save computation, we only process the upper right part of the matrix
                            if j_in_video > i:
                                save_images_index.append(j_in_video)
                                set_images.append(file_names[j_in_video])
                                captions_in_set.append(captions_in_videos[j_in_video])

                        captions_in_set = list(set(captions_in_set))

                        # collect information of captions in set 
                        if not len(captions_in_set) == 0:
                            for caption in captions_in_set:
                                caption_info = captions_labels[caption]
                                try:
                                    del caption_info["images"]
                                    in_captions_labels.update({caption:caption_info})
                                except KeyError:
                                    in_captions_labels.update({caption:caption_info})
                            caption_dict = {"images":set_images, "captions":in_captions_labels}
                            video_images_captions.update({set_id:caption_dict})

            # save information in videos
            with open(save_path, 'w') as fa:
                json.dump(video_images_captions, fa, indent=2)
