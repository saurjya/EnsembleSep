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
    dataset_name = "BBCSO"

    def __init__(self, json_file, n_src=4, sample_rate=22050, segment=220500, batch_size=1, train=False):
        super(BBCSODataset, self).__init__()
        # Task setting
        self.json_file = json_file
        self.sample_rate = sample_rate
        self.n_src = n_src
        self.batch_size = batch_size
        self.train = train
        #self.segment = int(segment)
        self.segment = 220500
        
        
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
            s, sr = sf.read(src, start=start, stop=stop, dtype="float32", always_2d=True)
            #s = np.zeros((self.segment,))
            s = s.mean(axis=1)
            #sr = 44100
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
    title="BBCSO-RWC Chamber Music Dataset: A Multitrack Dataset",
    author="S. Sarkar, M. Pilataki, D. Foster, E. Benetos and M. B. Sandler",
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License",
    license_link="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    non_commercial=True,
)
