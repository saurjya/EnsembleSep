import torch
from torch.utils import data
import json
import os
import numpy as np
import random
import soundfile as sf
import random
import torchaudio


class BBCSODataset(data.Dataset):
    """EnsembleSet Chamber Ensemble Separation Dataset
    
    This dataloader enables generating mixtures of N number of sources
    from the EnsembleSet dataset for a given list of instrument tags. This dataloader
    can be used to train PIT based separation models.
    
    The EnsembleSet dataset is hosted on Zenodo.
    https://zenodo.org/record/6519024
    """
    dataset_name = "BBCSO"

    def __init__(self, json_file, n_src=4, sample_rate=22050, segment=220500, batch_size=1, train=False, val=False):
        super(BBCSODataset, self).__init__()
        # Task setting
        self.json_file = json_file
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.batch_size = batch_size
        self.train = train
        #self.segment = int(segment)
        self.segment = 131072
        self.mixes = ['Amb','AtmosF','AtmosR','Balcony','Close','CloseW','Leader','Mids',
            'Mix_1','Mix_2','Mono','Out','Sides','SpBr','SpFl','SpPer','SpStr','SpWW','Stereo','Tree']
        #self.mixes = ['Close','CloseW','Leader','Mids',
        #    'Mix_1','Mix_2','Mono','Stereo','Tree']

        with open(json_file, "r") as f:
            sources_infos = json.load(f)
        
        self.sources = np.array(sources_infos)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        source_arrays = []
        start = int(self.sources[idx][-1])
        stop = start + self.segment
        for src in self.sources[idx][:self.n_src]:
            if self.train:
                if not self.val:
                    split = src.split('/')
                    mix = random.choice(self.mixes)
                    split[-2] = mix
                    src = '/'.join(split)
            s, sr = sf.read(src, start=start, stop=stop, dtype="float32", always_2d=True)
            s = s.mean(axis=1)
            source_arrays.append(s)
        source = torch.from_numpy(np.vstack(source_arrays))
        
        if sr is not self.sample_rate:
            source = torchaudio.transforms.Resample(sr, self.sample_rate)(source)

        if self.train:
            return source
        else:
            mix = torch.stack(list(source)).sum(0)
            return mix, source

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "chamber_sep_eval"
        infos["licenses"] = [BBCSO_license]
        return infos




BBCSO_license = dict(
    title="EnsembleSet: A new high quality dataset for chamber ensemble separation",
    author="S. Sarkar, E. Benetos and M. B. Sandler",
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License",
    license_link="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    non_commercial=True,
)
