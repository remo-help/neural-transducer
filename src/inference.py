import torch
from tqdm import tqdm
from typing import Dict, List, Optional
import numpy as np
import pickle as pkl
import util
import trainer
import dataloader
import decoding
import loader

args = {
        'model':'/home/ubuntu/transducer-rework/neural-transducer/checkpoints/transformer/transformer/transformer-dene0.3/latin-high-.nll_0.8365.acc_90.2491.dist_0.0521.epoch_85',
        'vocab': '/home/ubuntu/transducer-rework/neural-transducer/checkpoints/transformer/transformer/transformer-dene0.3/latin-high-.vocab',
        'file': '/home/ubuntu/transducer-rework/neural-transducer/data/latin-dev',
        'mode': 'testing'}

loader = loader.Loader(cli=False, config=args)

loader.run()
