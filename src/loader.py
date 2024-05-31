import os.path

import torch
from tqdm import tqdm
from typing import Dict, List, Optional
from math import ceil
import numpy as np
import pickle as pkl
from . import util

from . import dataloader
from . import decoding
import argparse

# we need this because loading a pytorch model we need to reference the model class in the same folder,
# so if we want to load a model from a higher working directory, we need to make sure pytorch can fidn the path
import sys
from pathlib import Path
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            full = os.path.join(root, name)
            dir = os.path.dirname(full)
            return dir



BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"
ALIGN = "<a>"
STEP = "<step>"
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
STEP_IDX = 4


class Loader:

    def __init__(self, cli=True, config=None):
        '''
        Initialize this class with cli=False to pass a config dictionary instead of using sys.var for the arguments
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.parser = argparse.ArgumentParser()
        self.cli = cli

        pth = find("transformer.py", Path.cwd())
        if pth:
            print(pth)
            sys.path.insert(0, pth)

        if not cli:
            self.set_args()
            config_list = []
            if config['file'] is not str:
                for key in config:
                    if key != 'file':
                        config_list.extend(f'--{key} {config[key]}'.split())
                        config_list.extend(f'--file placeholder'.split())
                self.params = self.parser.parse_args(config_list)
                self.params.file = config['file']
            else:
                for key in config:
                    config_list.extend(f'--{key} {config[key]}'.split())
                self.params = self.parser.parse_args(config_list)
        else:
            self.set_args()
            self.params = self.get_params()

        self.load_model(self.params.model)

        self.mode = True if self.params.mode == 'inference' else False

        if self.params.vocab:
            self.vocab_file = self.params.vocab
        else:
            if os.path.isfile(self.params.model + 'vocab'):
                self.vocab_file = self.params.model + 'vocab'
            else:
                self.vocab_file = None
        if self.params.datatype == 'direct':
            self.data = DirectLoader(self.params.file, load_vocab=self.vocab_file, inference_mode=self.mode,
                                     cli=self.cli)
        else:
            self.data = TabSeparated(self.params.file, load_vocab=self.vocab_file, inference_mode=self.mode,
                                    )

        if self.params.decode_fn == 'beam':
            self.decoder = decoding.Decoder(decoder_type=decoding.Decode.beam, max_len=30, beam_size=3)
        else:
            self.decoder = decoding.Decoder(decoder_type=decoding.Decode.greedy)

        self.batch_size = self.params.batch_size
        self.n_batches = ceil(self.data.nb_file / self.batch_size)

        self.logger = util.get_logger(
            self.params.model + ".log", log_level='info'
        )

        pass

    def set_args(self):
        """
        get_args
        """
        # fmt: off
        parser = self.parser
        parser.add_argument('--file', required=True, type=str, help="The path to the input file used for inference")
        parser.add_argument('--model', required=True, type=str, help="The path to the model checkpoint")
        parser.add_argument('--decode_fn', required=False, type=str, default='beam', help="Type of decode function")
        parser.add_argument('--mode', required=False, type=str, default='inference',
                            help="Load for inference or testing", choices=['inference', 'testing'])
        parser.add_argument('--vocab', required=False, type=str, help="The path to the vocab file, defaults"
                                                                      "to model path + .vocab", default=None)

        parser.add_argument('--batch_size', required=False, type=int, help="batch_size for inference",
                            default=64)
        parser.add_argument('--datatype', required=False, type=str, help="The type of data being read",
                                                                      default=None, choices= ['tabseparated', 'direct'])

    def get_params(self):
        return self.parser.parse_args()

    def load_model(self, path):
        assert self.model is None
        #self.logger.info("load model in %s", model)
        self.model = torch.load(path, map_location=self.device)
        self.model = self.model.to(self.device)
        print(type(self.model))

    def load_data(self):
        # do_something
        pass

    def testing(self):
        self.model.eval()
        self.logger.info('Running testing')
        with open(f"{self.params.model}.test.tsv", "w") as fp:
            fp.write("prediction\ttarget\tloss\tdist\n")
            avg_distance = []
            norm_distance = []
            accurates = 0
            totals = 0
            for src, src_mask, trg, trg_mask in tqdm(self.data.batch_sample(batch_size=self.batch_size),
                                                     total=self.n_batches):
                pred, _ = self.decoder(self.model, src, src_mask)
                #self.evaluator.add(src, pred, trg)

                #data = (src, src_mask, trg, trg_mask)
                #losses = self.model.get_loss(data, reduction=False).cpu()

                pred = util.unpack_batch(pred)
                trg = util.unpack_batch(trg)
                for p, t in zip(pred, trg):
                    dist = util.edit_distance(p, t)
                    norm_distance.append(dist / len(t))
                    avg_distance.append(dist)
                    p = self.data.decode_target(p)
                    t = self.data.decode_target(t)
                    if p == t:
                        accurates += 1
                    totals += 1
                    fp.write(f'{" ".join(p)}\t{" ".join(t)}\t{dist}\n')
            avg_distance = np.mean(np.array(avg_distance))
            norm_distance = np.mean(np.array(norm_distance))
            accuracy = accurates / totals
            self.logger.info(f"Average edit distance: {avg_distance}")
            self.logger.info(f"Normalized edit distance: {norm_distance}")
            self.logger.info(f"Average accuracy: {accuracy}")
            fp.write("\n ------------------------------------------------------------- \n")
            fp.write(f"Average edit distance: {avg_distance}\n")
            fp.write(f"Normalized edit distance: {norm_distance}\n")
            fp.write(f"Average accuracy: {accuracy}")

    def inference(self):
        # infer on loaded data, return results
        self.model.eval()
        results = []
        self.logger.info('Running inference')
        for src, mask in tqdm(self.data.batch_sample(batch_size=self.batch_size), total=self.n_batches):
            res, _ = self.decoder(self.model, src, mask)
            pred = util.unpack_batch(res)
            decodes = []
            for p in pred:
                p = ''.join(self.data.decode_target(p))
                decodes.append(p)
            results.extend(decodes)
        self.results = results
        self.save_results_inference()
        return results

    def save_results_inference(self):
        with open(f"{self.params.file}.{self.params.mode}.tsv", "w") as fp:
            for prediction in self.results:
                fp.write(f"{prediction}\n")

    def run(self):
        if self.mode:
            return self.inference()

        else:
            self.testing()
            return None


class InferenceDataloader(dataloader.Dataloader):
    def __init__(
            self,
            file: List[str],
            test_file: Optional[List[str]] = None,
            shuffle=False,
            load_vocab=None,
            inference_mode=True,
            cli = True
    ):
        super().__init__()
        self.file = file[0] if len(file) == 1 else file
        self.test_file = (
            test_file[0] if test_file and len(test_file) == 1 else test_file
        )
        self.shuffle = shuffle
        self.batch_data: Dict[str, List] = dict()
        self.nb_file, self.nb_test = 0, 0
        self.nb_attr = 0
        self.source, self.target = self.build_vocab(path=load_vocab)
        self.source_vocab_size = len(self.source)
        self.target_vocab_size = len(self.target)
        self.inference_mode = inference_mode
        self.attr_c2i: Optional[Dict]
        self.cli = cli
        if self.nb_attr > 0:
            self.source_c2i = {c: i for i, c in enumerate(self.source[: -self.nb_attr])}
            self.attr_c2i = {
                c: i + len(self.source_c2i)
                for i, c in enumerate(self.source[-self.nb_attr:])
            }
        else:
            self.source_c2i = {c: i for i, c in enumerate(self.source)}
            self.attr_c2i = None
        self.target_c2i = {c: i for i, c in enumerate(self.target)}
        self.sanity_check()

    def sanity_check(self):
        assert self.source[PAD_IDX] == PAD
        assert self.target[PAD_IDX] == PAD
        assert self.source[BOS_IDX] == BOS
        assert self.target[BOS_IDX] == BOS
        assert self.source[EOS_IDX] == EOS
        assert self.target[EOS_IDX] == EOS
        assert self.source[UNK_IDX] == UNK
        assert self.target[UNK_IDX] == UNK

    def build_vocab(self, path: str = None):
        if path:
            with open(path, 'rb') as f:
                vocabs = pkl.load(f)
                source = vocabs['source']
                target = vocabs['target']
            self.nb_file = 0
            for _ in self.read_file(self.file):
                self.nb_file += 1
        else:
            src_set, trg_set = set(), set()
            self.nb_file = 0
            for src, trg in self.read_file(self.file):
                self.nb_file += 1
                src_set.update(src)
                trg_set.update(trg)
            if self.test_file is not None:
                self.nb_test = sum([1 for _ in self.read_file(self.test_file)])
            source = [PAD, BOS, EOS, UNK] + sorted(list(src_set))
            target = [PAD, BOS, EOS, UNK] + sorted(list(trg_set))
        return source, target

    def store_vocab(self, path: str):
        vocabs = {'source': self.source, 'target': self.target}
        util.maybe_mkdir(path)
        with open(path, 'wb+') as f:
            pkl.dump(vocabs, f)

    def read_file(self, file, inference=True):
        raise NotImplementedError

    def _file_identifier(self, file):
        return file

    def list_to_tensor(self, lst: List[List[int]], max_seq_len=None):
        max_len = max([len(x) for x in lst])
        if max_seq_len is not None:
            max_len = min(max_len, max_seq_len)
        data = torch.zeros((max_len, len(lst)), dtype=torch.long)
        for i, seq in tqdm(enumerate(lst), desc="build tensor"):
            data[: len(seq), i] = torch.tensor(seq)
        mask = (data > 0).float()
        return data, mask

    def _batch_sample(self, batch_size, file, shuffle):
        if self.inference_mode:
            if not self.cli:
                key = 'placeholder'
            else:
                key = self._file_identifier(file)
            if key not in self.batch_data:
                lst = list()
                for src in tqdm(self._iter_helper(file), desc="read file"):
                    lst.append((src))
                src_data, src_mask = self.list_to_tensor([src for src in lst])

                self.batch_data[key] = (src_data, src_mask)

            src_data, src_mask = self.batch_data[key]
            nb_example = len(src_data[0])
            if shuffle:
                idx = np.random.permutation(nb_example)
            else:
                idx = np.arange(nb_example)
            for start in range(0, nb_example, batch_size):
                idx_ = idx[start: start + batch_size]
                src_mask_b = src_mask[:, idx_]
                src_len = int(src_mask_b.sum(dim=0).max().item())
                src_data_b = src_data[:src_len, idx_].to(self.device)
                src_mask_b = src_mask_b[:src_len].to(self.device)
                yield (src_data_b, src_mask_b)
        else:
            key = self._file_identifier(file)
            if key not in self.batch_data:
                lst = list()
                for src, trg in tqdm(self._iter_helper(file), desc="read file"):
                    lst.append((src, trg))
                src_data, src_mask = self.list_to_tensor([src for src, _ in lst])
                trg_data, trg_mask = self.list_to_tensor([trg for _, trg in lst])
                self.batch_data[key] = (src_data, src_mask, trg_data, trg_mask)

            src_data, src_mask, trg_data, trg_mask = self.batch_data[key]
            nb_example = len(src_data[0])
            if shuffle:
                idx = np.random.permutation(nb_example)
            else:
                idx = np.arange(nb_example)
            for start in range(0, nb_example, batch_size):
                idx_ = idx[start: start + batch_size]
                src_mask_b = src_mask[:, idx_]
                trg_mask_b = trg_mask[:, idx_]
                src_len = int(src_mask_b.sum(dim=0).max().item())
                trg_len = int(trg_mask_b.sum(dim=0).max().item())
                src_data_b = src_data[:src_len, idx_].to(self.device)
                trg_data_b = trg_data[:trg_len, idx_].to(self.device)
                src_mask_b = src_mask_b[:src_len].to(self.device)
                trg_mask_b = trg_mask_b[:trg_len].to(self.device)
                yield (src_data_b, src_mask_b, trg_data_b, trg_mask_b)

    def batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.file, shuffle=self.shuffle)

    def test_batch_sample(self, batch_size):
        yield from self._batch_sample(batch_size, self.test_file, shuffle=False)

    def encode_source(self, sent):
        if sent[0] != BOS:
            sent = [BOS] + sent
        if sent[-1] != EOS:
            sent = sent + [EOS]
        seq_len = len(sent)
        s = []
        for x in sent:
            if x in self.source_c2i:
                s.append(self.source_c2i[x])
            else:
                s.append(self.attr_c2i[x])
        return torch.tensor(s, device=self.device).view(seq_len, 1)

    def decode_source(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.source[x] for x in sent]

    def decode_target(self, sent):
        if isinstance(sent, torch.Tensor):
            assert sent.size(1) == 1
            sent = sent.view(-1)
        return [self.target[x] for x in sent]

    def _sample(self, file):
        for src, trg in self._iter_helper(file):
            yield (
                torch.tensor(src, device=self.device).view(len(src), 1),
                torch.tensor(trg, device=self.device).view(len(trg), 1),
            )

    def train_sample(self):
        yield from self._sample(self.train_file)

    def dev_sample(self):
        yield from self._sample(self.dev_file)

    def test_sample(self):
        yield from self._sample(self.test_file)

    def _iter_helper(self, file):
        if self.inference_mode:
            for source in self.read_file(file, inference=self.inference_mode):
                src = [self.source_c2i[BOS]]
                for s in source:
                    src.append(self.source_c2i.get(s, UNK_IDX))
                src.append(self.source_c2i[EOS])
                yield src
        else:
            for source, target in self.read_file(file, inference=self.inference_mode):
                src = [self.source_c2i[BOS]]
                for s in source:
                    src.append(self.source_c2i.get(s, UNK_IDX))
                src.append(self.source_c2i[EOS])
                trg = [self.target_c2i[BOS]]
                for t in target:
                    trg.append(self.target_c2i.get(t, UNK_IDX))
                trg.append(self.target_c2i[EOS])
                yield src, trg


class TabSeparated(InferenceDataloader):
    def read_file(self, file, inference=True):
        if inference:
            with open(file, "r", encoding="utf-8") as fp:
                for line in fp.readlines():
                    X, _ = line.strip().split("\t")
                    yield list(X)
        else:
            with open(file, "r", encoding="utf-8") as fp:
                for line in fp.readlines():
                    X, y = line.strip().split("\t")
                    yield list(X), list(y)


class DirectLoader(InferenceDataloader):
    '''
    Used to pass data directly, instead of loading it from a file.
    '''

    def read_file(self, file, inference=True):
        if inference:
            for item in file:
                yield [item]
        else:
            for item in file:
                X, y = item
                yield list(X), list(y)


if __name__ == "__main__":
    loader = Loader(cli=True)

    loader.run()
