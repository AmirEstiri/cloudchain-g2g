import cv2
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob, os
from tqdm import tqdm
from keras.preprocessing import image


class Dataloader:
    def __init__(self, path='../data/UCF-101/', frames_path='../data/UCF-101-train-1/'):
        self.path = path
        self.frames_path = frames_path
        self.videos = pd.DataFrame()


    def load_video_names(self):
        video_paths = []
        video_names = []
        video_class = []
        for path, subdirs, files in os.walk(self.path):
            for name in files:
                video_class.append(path.split('/')[-1])
                video_paths.append(os.path.join(path, name))
                video_names.append(name.split('.')[0])
        self.videos['path'] = video_paths
        self.videos['name'] = video_names
        self.videos['class'] = video_class
        # self.videos = self.videos.sample(frac=1)


    def make_dirs(self):
        classes = np.unique(self.videos['class'])
        os.mkdir(self.frames_path)
        for dir in classes:
            os.mkdir(self.frames_path+dir)


    def create_frames(self):
        for i in tqdm(range(len(self.videos))):
            count = 0
            vid = cv2.VideoCapture(self.videos.iloc[i]['path'])
            frame_rate = vid.get(5)
            while(vid.isOpened()):
                frame_id = vid.get(1)
                ret, frame = vid.read()
                if (not ret):
                    break
                if (frame_id % math.floor(frame_rate) == 0):
                    filename = self.frames_path + self.videos.iloc[i]['class'] + '/' + self.videos.iloc[i]['name'] +"_frame%d.jpg" % count;count+=1
                    cv2.imwrite(filename, frame)
            vid.release()


    def create_data(self, start, end):
        image_names = []
        dir_names = []
        frames = pd.DataFrame()
        for path, subdirs, files in os.walk(self.frames_path):
            for name in files:
                dir_names.append(path.split('/')[-1])
                image_names.append(name)
        frames['name'] = image_names
        frames['dir'] = dir_names
        classes = np.unique(frames['dir'])
        X = []
        for k in range(start, end):
            for i in tqdm(range(len(frames[frames['dir']==classes[k]]))):
                img = image.load_img(self.frames_path + frames.iloc[i]['dir'] + '/' + frames.iloc[i]['name'], target_size=(224,224,3))
                img = image.img_to_array(img)
                img = img/255
                X.append(img)
        X = np.array(X)
        x_train, x_val, _, __ = train_test_split(X, X, random_state=42, test_size=0.1)
        return x_train, x_val

