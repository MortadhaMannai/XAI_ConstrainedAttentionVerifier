import pickle
import sys
from os import path

import pandas as pd
import pytorch_lightning as pl
import spacy
import torch
import torchtext.transforms as T
from data import transforms as t
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator, GloVe
from tqdm import tqdm
from transformers import BertTokenizer

from data.esnli.pretransform_dataset import PretransformedESNLI
from data.esnli.transforms import HighlightTransform, HeuristicTransform
from data.transforms import LemmaLowerTokenizerTransform
from modules import env, INF
from modules.const import SpecToken, Normalization
from modules.logger import log


class ESNLIDM(pl.LightningDataModule):
    name = 'eSNLI'

    def __init__(self, cache_path, batch_size=8, num_workers=0, n_data=-1, shuffle=True):
        super().__init__()
        self.cache_path = cache_path
        self.batch_size = batch_size
        # Dataset already tokenized
        self.n_data = n_data
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.input_type = PretransformedESNLI.INPUT
        self.LABEL_ITOS = PretransformedESNLI.LABEL_ITOS

        spacy_model = spacy.load('en_core_web_sm')
        tokenizer_transform = LemmaLowerTokenizerTransform(spacy_model)
        hl_transform = T.Sequential(tokenizer_transform, HighlightTransform())
        heuristic_transform = HeuristicTransform(
            vectors=GloVe(cache=path.join(cache_path, '..', '.vector_cache')),
            spacy_model=spacy_model,
            cache=cache_path
        )

        self.transformations = {
            'premise': tokenizer_transform,
            'hypothesis': tokenizer_transform,
            'highlight_premise': hl_transform,
            'highlight_hypothesis': hl_transform,
            'heuristic': heuristic_transform,
        }
        self.rename_columns = {
            'premise': 'premise_tokens',
            'hypothesis': 'hypothesis_tokens',
            'highlight_premise': 'premise_rationale',
            'highlight_hypothesis': 'hypothesis_rationale',
            'heuristic': 'heuristic'
        }

    def prepare_data(self):
        # called only on 1 GPU

        # download_dataset()
        dataset_path = PretransformedESNLI.root(self.cache_path)
        self.vocab_path = path.join(dataset_path, f'vocab.pt')

        for split in ['train', 'val', 'test']:
            PretransformedESNLI.download_format_dataset(dataset_path, split)

        # build_vocab()
        if not path.exists(self.vocab_path):

            # return a single list of tokens
            def flatten_token(batch):
                return [token for sentence in batch['premise_tokens'] + batch['hypothesis_tokens'] for token in
                        sentence]

            train_set = PretransformedESNLI(transformations=self.transformations, column_name=self.rename_columns, root=self.cache_path, split='train', n_data=self.n_data)

            # build vocab from train set
            dp = train_set.batch(self.batch_size).map(self.list2dict).map(flatten_token)

            # Build vocabulary from iterator. We don't know yet how long does it take
            iter_tokens = tqdm(iter(dp), desc='Building vocabulary', total=len(dp), unit='sents', file=sys.stdout,
                               disable=env.disable_tqdm)
            if env.disable_tqdm: log.info(f'Building vocabulary')
            vocab = build_vocab_from_iterator(iterator=iter_tokens, specials=[SpecToken.PAD, SpecToken.UNK])

            vocab.set_default_index(vocab[SpecToken.UNK])

            # Announce where we save the vocabulary
            torch.save(vocab, self.vocab_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)  # Use highest protocol to speed things up
            iter_tokens.set_postfix({'fpath': self.vocab_path})
            if env.disable_tqdm: log.info(f'Vocabulary is saved at {self.vocab_path}')
            iter_tokens.close()
            self.vocab = vocab
        else:
            self.vocab = torch.load(self.vocab_path)
            log.info(f'Loaded vocab at {self.vocab_path}')

        log.info(f'Vocab size: {len(self.vocab)}')

        # Clean cache
        PretransformedESNLI.clean_cache(root=self.cache_path)

        # predefined processing mapper for setup
        self.text_transform = T.Sequential(
            T.VocabTransform(self.vocab),
            T.ToTensor(padding_value=self.vocab[SpecToken.PAD])
        )

        self.rationale_transform = T.Sequential(
            t.ToTensor(padding_value=0)
        )

        self.heuristic_transform = T.Sequential(
            t.ToTensor(padding_value=-INF, dtype=torch.float32),
            t.NormalizationTransform(normalize=Normalization.LOG_SOFTMAX)
        )

        self.label_transform = T.Sequential(
            T.LabelToIndex(['neutral', 'entailment', 'contradiction']),
            T.ToTensor()
        )

    def setup(self, stage: str = None):
        dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data, transformations=self.transformations, column_name=self.rename_columns)

        # called on every GPU
        if stage == 'fit' or stage is None:
            self.train_set = PretransformedESNLI(split='train', **dataset_kwargs)
            self.val_set = PretransformedESNLI(split='val', **dataset_kwargs)

        if stage == 'test' or stage == 'predict' or stage is None:
            self.test_set = PretransformedESNLI(split='test', **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()

    def format_predict(self, prediction: pd.DataFrame):
        
        # replace label
        label_columns = ['y_hat', 'y_true']
        label_itos = {idx: val for idx, val in enumerate(self.LABEL_ITOS)}
        prediction.replace({c: label_itos for c in label_columns}, inplace=True)
        
        return prediction

    ## ======= PRIVATE SECTIONS ======= ##

    def collate(self, batch):
        # prepare batch of data for dataloader
        b = self.list2dict(batch)

        b.update({
            'premise_ids': self.text_transform(b['premise_tokens']),
            'hypothesis_ids': self.text_transform(b['hypothesis_tokens']),
            'a_true': {
                'premise': self.rationale_transform(b['premise_rationale']),
                'hypothesis': self.rationale_transform(b['hypothesis_rationale']),
            },
            'y_true': self.label_transform(b['label']),
            'heuristic' : {
                'premise': self.heuristic_transform(b['premise_heuristic']),
                'hypothesis': self.heuristic_transform(b['hypothesis_heuristic']),
            }
        })

        b['padding_mask'] = {
            'premise': b['premise_ids'] == self.vocab[SpecToken.PAD],
            'hypothesis': b['hypothesis_ids'] == self.vocab[SpecToken.PAD],
        }
        
        return b

    def list2dict(self, batch):
        # convert list of dict to dict of list

        if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}  # handle case where no batch
        return {k: [row[k] for row in batch] for k in batch[0]}

class BertESNLIDM(pl.LightningDataModule):
    name = 'eSNLI'

    def __init__(self, cache_path, batch_size=8, num_workers=0, n_data=-1, shuffle=True):
        super().__init__()
        self.cache_path = cache_path
        self.batch_size = batch_size
        # Dataset already tokenized
        self.n_data = n_data
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.num_class = PretransformedESNLI.NUM_CLASS
        self.input_type = PretransformedESNLI.INPUT

        spacy_model = spacy.load('en_core_web_sm')
        
        tokenizer_transform = BertTokenizer.from_pretrained("bert-base-uncased")
        
        hl_transform = T.Sequential(tokenizer_transform, HighlightTransform())
        heuristic_transform = HeuristicTransform(
            vectors=GloVe(cache=path.join(cache_path, '..', '.vector_cache')),
            spacy_model=spacy_model,
            normalize='log_softmax'
        )

        self.transformations = {
            'premise': tokenizer_transform,
            'hypothesis': tokenizer_transform,
            'highlight_premise': hl_transform,
            'highlight_hypothesis': hl_transform,
            'heuristic': heuristic_transform,
        }
        self.rename_columns = {
            'premise': 'premise_tokens',
            'hypothesis': 'hypothesis_tokens',
            'highlight_premise': 'premise_rationale',
            'highlight_hypothesis': 'hypothesis_rationale',
            'heuristic': 'heuristic'
        }

    def prepare_data(self):
        # called only on 1 GPU

        # download_dataset()
        dataset_path = PretransformedESNLI.root(self.cache_path)
        self.vocab_path = path.join(dataset_path, f'bert_vocab.pt')

        for split in ['train', 'val', 'test']:
            PretransformedESNLI.download_format_dataset(dataset_path, split)

        log.info(f'Vocab size: {len(self.vocab)}')

        # Clean cache
        PretransformedESNLI.clean_cache(root=self.cache_path)

        # predefined processing mapper for setup
        self.text_transform = T.Sequential(
            T.VocabTransform(self.vocab),
            T.ToTensor(padding_value=self.vocab[SpecToken.PAD])
        )

        self.rationale_transform = T.Sequential(
            T.ToTensor(padding_value=0)
        )

        self.label_transform = T.Sequential(
            T.LabelToIndex(PretransformedESNLI.LABEL_ITOS),
            T.ToTensor()
        )

    def setup(self, stage: str = None):
        dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data, transformations=self.transformations, column_name=self.rename_columns, )

        # called on every GPU
        if stage == 'fit' or stage is None:
            self.train_set = PretransformedESNLI(split='train', **dataset_kwargs)
            self.val_set = PretransformedESNLI(split='val', **dataset_kwargs)

        if stage == 'test' or stage == 'predict' or stage is None:
            self.test_set = PretransformedESNLI(split='test', **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()

    ## ======= PRIVATE SECTIONS ======= ##

    def collate(self, batch):
        # prepare batch of data for dataloader
        b = self.list2dict(batch)

        b.update({
            'premise_ids': self.text_transform(b['premise_tokens']),
            'hypothesis_ids': self.text_transform(b['hypothesis_tokens']),
            'a_true': {
                'premise': self.rationale_transform(b['premise_rationale']),
                'hypothesis': self.rationale_transform(b['hypothesis_rationale']),
            },
            'y_true': self.label_transform(b['label'])
        })

        b['padding_mask'] = {
            'premise': b['premise_ids'] == self.vocab[SpecToken.PAD],
            'hypothesis': b['hypothesis_ids'] == self.vocab[SpecToken.PAD],
        }

        return b

    def list2dict(self, batch):
        # convert list of dict to dict of list

        if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}  # handle case where no batch
        return {k: [row[k] for row in batch] for k in batch[0]}

class CLSTokenESNLIDM(ESNLIDM):

    def __init__(self, **kwargs):
        super(CLSTokenESNLIDM, self).__init__(**kwargs)

    def prepare_data(self):
        super(CLSTokenESNLIDM, self).prepare_data()
    
        # called only on 1 GPU
        VOCAB_SUFFIX = '_CLS'
        if VOCAB_SUFFIX not in self.vocab_path:
            fname, fext = path.splitext(self.vocab_path)
            self.vocab_path = fname + VOCAB_SUFFIX + fext
    
        # build_vocab()
        if not path.exists(self.vocab_path):
            self.vocab.append_token(SpecToken.CLS)
        
            # Announce where we save the vocabulary
            torch.save(self.vocab, self.vocab_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)  # Use highest protocol to speed things up
            if env.disable_tqdm: log.info(f'Vocab CLS is saved at {self.vocab_path}')
    
        else:
            self.vocab = torch.load(self.vocab_path)
            log.info(f'Loaded vocab CLS at {self.vocab_path}')
    
        log.info(f'Vocab size: {len(self.vocab)}')
        
        
    def collate(self, batch):
        b = super(CLSTokenESNLIDM, self).collate(batch)

        # we change here the shape of the dictionary b
        # concatenation for one sentence.
        t = b["premise_ids"].shape[0]
        cls_ids = torch.tensor([self.vocab[SpecToken.CLS]]).repeat(t, 1)
        cls_padding = torch.tensor([0.]).repeat(t, 1)
        att_padding = torch.tensor([0.]).repeat(t, 1)
        num = torch.log(b['a_true']['premise'].sum(dim=-1) + b['a_true']['hypothesis'].sum(dim=-1))

        den = torch.log(
            (~b['padding_mask']['premise']).sum(dim=-1) + (~b['padding_mask']['hypothesis']).sum(dim=-1)
        )

        b.update({
            'token_ids': torch.cat((cls_ids, b['premise_ids'], b['hypothesis_ids']), 1),
            'padding_mask': torch.cat((cls_padding, b['padding_mask']['premise'], b['padding_mask']['hypothesis']), 1),
            'a_true': torch.cat((att_padding, b['a_true']['premise'], b['a_true']['hypothesis']), 1),
            'a_true_entropy': num / den
        })

        return b
