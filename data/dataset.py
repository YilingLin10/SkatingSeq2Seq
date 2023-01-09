import numpy as np
import os 
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import glob
import cv2
from pathlib import Path
import pickle
import json

class IceSkatingDataset(Dataset):

    def __init__(self, pkl_file, tag_mapping_file, subtract_feature, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(pkl_file, 'rb') as f:
            self.video_data_list = pickle.load(f)
        self.transform = transform
        self.subtract_feature = subtract_feature
        self.tag_mapping = json.loads(Path(tag_mapping_file).read_text())
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
    
    def __len__(self):
        return len(self.video_data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_name = self.video_data_list[idx]['video_name']
        tags = self.video_data_list[idx]['output']
        if self.subtract_feature:
            features_list = self.video_data_list[idx]['subtraction_features']
        else:
            features_list = self.video_data_list[idx]['features']
        sample = {"keypoints": torch.FloatTensor(features_list), "video_name": video_name, "output": torch.LongTensor(tags)}
        return sample
    
    def collate_fn(self, samples):
        PAD_IDX = 4
        video_names, src_batch, tgt_batch = [], [], []
        for sample in samples:
            video_names.append(sample["video_name"])
            src_batch.append(sample["keypoints"])
            tgt_batch.append(sample["output"])
        src_batch = pad_sequence(src_batch, padding_value=0)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return video_names, src_batch, tgt_batch
    
    def tag2idx(self, tag: str):
        return self.tag_mapping[tag]

    def idx2tag(self, idx: int):
        return self._idx2tag[idx]

if __name__ == '__main__':
    dataset = IceSkatingDataset(pkl_file='/home/lin10/projects/SkatingJumpClassifier/data/flip/alphapose/test.pkl',
                                tag_mapping_file='/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json',
                                subtract_feature=False)

    dataloader = DataLoader(dataset,batch_size=2,
                        shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for i_batch, (video_names, src, tgt) in enumerate(dataloader):
        print(src.shape)
        print(tgt.shape)
        print(src)
        break