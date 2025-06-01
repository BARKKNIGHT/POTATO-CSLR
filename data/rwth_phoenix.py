import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import random
from PIL import Image
import json

class rwth_phoenix(Dataset):
    def __init__(self, csv, data_path, frame_transform, video_transform, input_fps, output_fps, max_frames, stride, word_dict):

        temp = np.loadtxt(csv, delimiter='|', dtype='str')
        self.csv = np.delete(temp, 0, 0)
        self.word_dict = word_dict
        self.data_path = data_path

        self.frame_transform = frame_transform
        self.video_transform = video_transform

        self.input_fps = input_fps
        self.output_fps = output_fps
        self.max_frames = max_frames
        self.stride = stride


    def __len__(self):
        return len(self.csv)


    def __getitem__(self, idx):
        folder = self.csv[idx][1].split('/')
        label = self.csv[idx][3].split(' ')

        words = []
        for word in label:
            words.append(self.word_dict[word])
        label = torch.tensor(words)
        image_folder_path = os.path.join(self.data_path, folder[0], folder[1])

        images = os.listdir(image_folder_path)
        end = len(images)

        step = self.input_fps/random.choice(self.output_fps)
        image_list = []
        if int(end//step +1) <= self.max_frames:
            frame_num, num = 0, 0
            while frame_num < end:
                num+=1
                if self.stride and num%self.stride == 0:
                    image_list.append(str(int(frame_num))+'a')

                else:
                    img = Image.open(os.path.join(image_folder_path, images[int(frame_num)]))
                    tensor_img = self.frame_transform(img)
                    image_list.append(tensor_img)
                frame_num += step

            c, h, w = image_list[-1].shape
            while len(image_list) < self.max_frames:
                image_list.append(torch.zeros(c,h,w))

            tensor_video = torch.stack(image_list[:self.max_frames])
            tensor_video = self.video_transform(tensor_video)

        else:
            frame_positions = np.linspace(0, end, self.max_frames, endpoint=False)
            num = 0
            for n in frame_positions:
                num+=1
                if self.stride and num%self.stride == 0:
                    image_list.append(str(int(n))+'a')

                else:
                    img = Image.open(os.path.join(image_folder_path, images[int(n)]))
                    tensor_img = self.frame_transform(img)
                    image_list.append(tensor_img)

            tensor_video = torch.stack(image_list[:self.max_frames])
            tensor_video = self.video_transform(tensor_video)


        return tensor_video, label