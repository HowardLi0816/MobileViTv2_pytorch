import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import io
from tqdm import tqdm
from ml_things import fix_text

class imdb_dataset(Dataset):

    def __init__(self, data_path, args, a_logger, split='train'):
        self.max_len = args.max_len

        self.split = split

        self.texts = []
        self.labels = []
        self.masks = []


        if split=='train':
            num_sample_used=args.train_sample_used
        elif split=='val':
            num_sample_used = args.val_sample_used
        elif split=='test':
            num_sample_used = args.test_sample_used

        dic = {'pos': 1, 'neg': 0}
        for label in ['pos', 'neg']:
            sentiment_path = os.path.join(data_path, label)

            files_names = os.listdir(sentiment_path)[:num_sample_used]

            for file_name in tqdm(files_names, desc=f'{label} files'):
                file_path = os.path.join(sentiment_path, file_name)

                # Read content
                content = io.open(file_path, mode='r', encoding='utf-8').read()
                # Fix any unicode issues
                content = fix_text(content)
                # Save content
                self.texts.append(content)
                # Save encode labels
                self.labels.append(dic[label])

        a_logger.info('Succesfully loaded {} dataset.'.format(split) + ' ' * 50)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.labels[idx]

        #if self.augmentation and self.transform != None:

        img=self.transform(img)

        #if self.feature_extractor:
        #    img = torch.squeeze(self.feature_extractor(img, return_tensors='pt')['pixel_values'])

        return img, target
