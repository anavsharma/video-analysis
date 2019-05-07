import sys
import argparse
import logging
import random
import os
from os import path
import json
import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

FRAME_PATH = "frames"
FEATURE_PATH = "features"
JSON_PATH = "json_data"

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video-path', type=str, required=True, help='path to the video file')
parser.add_argument('-o', '--output-directory', type=str, default=FRAME_PATH, required=False, help='path to the output directory')
parser.add_argument('-n', '--name', type=str, required=True, help='name of the video')
args = parser.parse_args()

def getFrames():
    '''
    Function to generate frames from a video file
    '''
    frameList = list()
    changePoints = list()
    picks = list()
    frames_per_seg = list()

    args.name.replace(' ', '_')
    fullOutputPath=path.join(args.output_directory, args.name)
    if not path.exists(fullOutputPath):
        print(fullOutputPath)
        os.mkdir(fullOutputPath)
    cap = cv2.VideoCapture(args.video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_steps = 0
    frameCounter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(fullOutputPath + "/%#06d.jpg" % (n_steps+1), frame)
        
        if frameCounter == n_steps:
            frameList.append(fullOutputPath + "/%#06d.jpg" % (n_steps+1))
            picks.append(n_steps+1)
            frameCounter += 10
        
        n_steps += 1

    print("In getFrames, value of frameCounter: "+str(frameCounter))
    cap.release()
    totalChangePoints = random.randint(1,15)
    changepointCompute = list()
    for i in range(totalChangePoints):
        changepointCompute.append(random.randint(30,frameCounter))
    changepointCompute = sorted(changepointCompute)
    
    for i in range(len(changepointCompute)):
        if i == 0:
            cpStart = 0
            cpEnd = changepointCompute[i]
        elif i == len(changepointCompute) - 1:
            cpStart = changepointCompute[i]
            cpEnd = frameCounter
        else:
            cpStart = changepointCompute[i-1]+1
            cpEnd = changepointCompute[i]
        changePoints.append([cpStart, cpEnd])
        frames_per_seg.append(cpEnd-cpStart)

    return frameList, n_frames, n_steps, picks, changePoints, frames_per_seg, fullOutputPath

class GoogLeNetPartial(nn.Module):
    def __init__(self,
                 layer='fc6',
                 model_file=None,
                 data_parallel=False,
                 **kwargs):         
        super(GoogLeNetPartial, self).__init__()
        self.model = models.googlenet(pretrained=True)
        self.output_layer = layer
        for param in self.model.parameters():
            param.requires_grad = False
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1024)
        self.input_size = 224

        features = list(self.model.children())[:17]
        # print(features)

        self.model.featExtract = nn.Sequential(*features)

    def forward(self, x):
        in_size = x.size(0)
        x = self.model.featExtract(x)
        x = x.view(in_size, -1)
        return x

class ListDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_list,
                 transform=None,
                 loader=default_loader):
        self.images_list = images_list
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.images_list[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.images_list)


def image_path_to_name(image_path):
    # return np.string_(path.splitext(path.basename(image_path))[0])
    parent, image_name = path.split(image_path)
    image_name = path.splitext(image_name)[0]
    parent = path.split(parent)[1]
    return path.join(parent, image_name)

def extract_features_to_disk(image_paths, n_frames, n_steps, picks, changePoints, frames_per_seg,
                             model,
                             batch_size,
                             workers,
                             output_hdf5):
    # Data loading code
    fullFeaturePath = os.path.join(FEATURE_PATH, output_hdf5)
    if not path.exists(FEATURE_PATH):
        os.mkdir(FEATURE_PATH)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = ListDataset(image_paths,
                          transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize,
                          ]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

    features = {}
    with torch.no_grad():
        for i, (input_data, paths) in enumerate(tqdm(loader)):
            input_var = torch.autograd.Variable(input_data)
            current_features = model(input_var).data.cpu().numpy()
            for j, image_path in enumerate(paths):
                features[image_path] = current_features[j]

    feature_shape = features[list(features.keys())[0]].shape
    logging.info('Feature shape: %s' % (feature_shape, ))
    logging.info('Outputting features')

    if sys.version_info >= (3, 0):
        string_type = h5py.special_dtype(vlen=str)
    paths = features.keys()
    logging.info('Stacking features')
    features_stacked = np.vstack([features[path] for path in paths])
    logging.info('Output feature size: %s' % (features_stacked.shape, ))
    key = args.name
    with h5py.File(fullFeaturePath, 'a') as f:
        print("-------------------------------------------")
        # 2D-array with shape (n_steps, feature-dimension)
        f.create_dataset(key+'/features', data=features_stacked)
        # 2D-array with shape (num_segments, 2), each row stores indices of a segment
        f.create_dataset(key+'/change_points', data=changePoints)
        print(changePoints)
        print("--")
        # 1D-array with shape (num_segments), indicates number of frames in each segment
        f.create_dataset(key+'/n_frame_per_seg', data=frames_per_seg)
        print(frames_per_seg)
        print("--")
        # number of frames in original video
        f.create_dataset(key+'/n_frames', data=n_frames)
        print(n_frames)
        print("--")
        # posotions of subsampled frames in original video
        f.create_dataset(key+'/picks', data=picks)
        print(picks)
        print("--")
        # number of subsampled frames
        f.create_dataset(key+'/n_steps', data=n_steps)
        print(n_steps)
        print("-------------------------------------------")

def _set_logging(logging_filepath):
    """Setup logger to log to file and stdout."""
    log_format = '%(asctime)s.%(msecs).03d: %(message)s'
    date_format = '%H:%M:%S'

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logging_filepath)
    file_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    logging.info('Writing log file to %s', logging_filepath)

def create_json_data():
    fullJsonPath = path.join(JSON_PATH,args.name)
    if not path.exists(fullJsonPath):
        print(fullJsonPath)
        os.mkdir(fullJsonPath)
    video_map = {}
    video_map[args.name] = args.name
    with open(fullJsonPath+'/video_map.json', 'w') as videoMapFile:
        json.dump(video_map, videoMapFile)
    
    splits = {}
    splits["train_keys"] = []
    splits["test_keys"] = [args.name]
    wrapper = list()
    wrapper.append(splits)
    with open(fullJsonPath+'/splits.json', 'w') as splitsFile:
        json.dump(wrapper, splitsFile)

def main():
    image_paths, n_frames, n_steps, picks, changePoints, frames_per_seg, fullOutputPath = getFrames()
    random.seed(0)
    torch.manual_seed(0)
    _set_logging('log/featureExtraction.log')
    construction_kwargs = {
        'layer': 'fc-6',
        'pretrained': True,
        'model_file': None,
        'data_parallel': True,
    }
    model = GoogLeNetPartial(**construction_kwargs)
    model.eval()

    extract_features_to_disk(image_paths, n_frames, n_steps, picks, changePoints, frames_per_seg, model, 10, 4, args.name+'_features.h5')
    create_json_data()

if __name__ == "__main__":
    main()