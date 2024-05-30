import torch
from tqdm import tqdm
from typing import Dict, List, Optional
import numpy as np
import pickle as pkl
from . import loader

if __name__ == "__main__":
        args = {}

        loader = loader.Loader(cli=False, config=args)

        loader.run()
