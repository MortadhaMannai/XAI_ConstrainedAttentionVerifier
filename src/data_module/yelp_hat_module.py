import pickle
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import spacy
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from data import transforms as t
from tqdm import tqdm
from os import path

from data.transforms import EntropyTransform
from data.yelp_hat.spacy_pretok_dataset import SpacyPretokenizeYelpHat
from modules import env
from modules.const import Normalization, SpecToken, INF
from modules.logger import log

class YelpHatDM(pl.LightningDataModule):
	
	name='YelpHat'
	
	def __init__(self, cache_path, batch_size=8, num_workers=0, n_data=-1, shuffle=True):
		super().__init__()
		self.cache_path = cache_path
		self.batch_size = batch_size
		
		# Dataset already tokenized
		self.n_data = n_data
		self.num_workers = num_workers
		self.spacy_model = spacy.load('en_core_web_sm')
		self.shuffle = shuffle
		self.input_type = SpacyPretokenizeYelpHat.INPUT
		self.LABEL_ITOS = SpacyPretokenizeYelpHat.LABEL_ITOS
		
		# configuration for predict writer
		self.predict_config = {
			'label_name' : 'y_hat',
			'html.token' : ['a_true', 'a_hat', 'heuristic']
		}
	
	def prepare_data(self):
		# called only on 1 GPU
		
		# Avoid preparing twice
		if hasattr(self, 'vocab') and self.vocab is not None: return
		
		# download_dataset()
		dataset_path = SpacyPretokenizeYelpHat.root(self.cache_path)
		lemma = True
		lower = True
		self.vocab_path = path.join(dataset_path, f'vocab{"_lemma" if lemma else ""}{"_lower" if lower else ""}.pt')
		
		SpacyPretokenizeYelpHat.download_format_dataset(dataset_path)  # only 1 line, download all dataset
		
		# build_vocab()
		if not path.exists(self.vocab_path):
			
			# return a single list of tokens
			def flatten_token(batch):
				return [token for sentence in batch['text_tokens'] for token in sentence]
			
			self.train_set = SpacyPretokenizeYelpHat(root=self.cache_path, split='train', n_data=self.n_data, lemma=lemma, lower=lower)
			
			# build vocab from train set
			dp = self.train_set.batch(self.batch_size).map(self.list2dict).map(flatten_token)
			
			# Build vocabulary from iterator. We don't know yet how long does it take
			iter_tokens = tqdm(iter(dp), desc='Building vocabulary', total=len(dp), unit='sents', file=sys.stdout, disable=env.disable_tqdm)
			if env.disable_tqdm: log.info(f'Building vocabulary')
			vocab = build_vocab_from_iterator(iterator=iter_tokens, specials=[SpecToken.PAD, SpecToken.UNK])
			# vocab = build_vocab_from_iterator(iter(doc for doc in train_set['post_tokens']), specials=[SpecToken.PAD, SpecToken.UNK])
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
		SpacyPretokenizeYelpHat.clean_cache(root=self.cache_path)
		
		# predefined processing mapper for setup
		self.text_transform = T.Sequential(
			T.VocabTransform(self.vocab),
			T.ToTensor(padding_value=self.vocab[SpecToken.PAD])
		)
		
		self.ham_transform = T.Sequential(
			T.ToTensor(padding_value=0)
		)
		
		self.label_transform = T.Sequential(
			# T.LabelToIndex(['0', '1']), # parquet convert already in int
			T.ToTensor()
		)
		
		self.entropy_transform = EntropyTransform()
		
		self.heuristic_transform = T.Sequential(
			t.ToTensor(padding_value=0, dtype=torch.float),
			t.ReplaceTransform(value=0, replace_by=-INF),
			t.NormalizationTransform(normalize=Normalization.LOG_SOFTMAX)
		)
		
	def setup(self, stage: str = None):
		dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data)
		
		# called on every GPU
		if stage == 'fit' or stage is None:
			if not hasattr(self, 'train_set'):
				self.train_set = SpacyPretokenizeYelpHat(split='train', **dataset_kwargs)
			if not hasattr(self, 'val_set'):
				self.val_set = SpacyPretokenizeYelpHat(split='val', **dataset_kwargs)
		
		if (stage == 'test' or stage is None) and not hasattr(self, 'test_sets'):
			self.test_sets = {key: SpacyPretokenizeYelpHat(split=key, **dataset_kwargs) for key in ['yelp50', 'yelp100', 'yelp200']}
	
	def train_dataloader(self):
		return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate, num_workers=self.num_workers)
	
	def val_dataloader(self):
		return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
	
	def test_dataloader(self):
		loader_kwargs = dict(batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
		# loaders = {dataset_name : DataLoader(dataset, **loader_kwargs) for dataset_name, dataset in self.test_sets.items()}
		loaders = [DataLoader(dataset, **loader_kwargs) for dataset_name, dataset in self.test_sets.items()]
		# return CombinedLoader(loaders, mode="max_size_cycle") # Run multiple test set in parallel
		return loaders
	
	def predict_dataloader(self):
		return self.test_dataloader()
	
	def format_predict(self, prediction: pd.DataFrame):
		
		# replace label
		label_columns = ['y_hat', 'y_true']
		label_itos = {idx: val for idx, val in enumerate(self.LABEL_ITOS)}
		prediction.replace({c: label_itos for c in label_columns}, inplace=True)
		
		# normalize heuristic into distribution
		sum_heuris = prediction['heuristic'].apply(lambda x: sum(x))
		if (sum_heuris != 1).any():
			prediction['heuristic'] =  prediction['heuristic'].apply(lambda x: np.exp(x).tolist())
		
		if 'text_tokens' not in prediction.columns:
			itos = self.vocab.get_itos()
			text_tokens = [[itos[ids] for ids in token_ids ] for token_ids in prediction['token_ids'].tolist()]
			prediction['text_tokens'] = pd.Series(text_tokens)
		
		return prediction
	
	## ======= PRIVATE SECTIONS ======= ##
	def collate(self, batch):
		# prepare batch of data for dataloader
		batch = self.list2dict(batch)

		b = {
			'token_ids': self.text_transform(batch['text_tokens']),
			'a_true': self.ham_transform(batch['ham']),
			'heuristic': self.heuristic_transform(batch['heuristic']),
			'y_true': self.label_transform(batch['label'])
		}
		
		b['padding_mask'] = b['token_ids'] == self.vocab[SpecToken.PAD]
		b['a_true_entropy'] = self.entropy_transform(b['a_true'], b['padding_mask'])
		return b
	
	def list2dict(self, batch):
		# convert list of dict to dict of list
		if isinstance(batch, dict): return {k: list(v) for k, v in batch.items()}  # handle case where no batch
		return {k: [row[k] for row in batch] for k in batch[0]}


class TransformedYelpHat50DM(YelpHatDM):
	#TODO working on this. If this work : replace YelpHat by new class
	
	name = 'YelpHat-50'
	
	def __init__(self, **kwargs):
		super(TransformedYelpHat50DM, self).__init__(**kwargs)
		
		for idx in range(3):
			data[f'ham_{idx}'] = data[f'ham_html_{idx}'].apply(lambda x: yelp_hat_ham(x, spacy_model)).apply(
				lambda x: np.array(x))
			
		binarize_ham_transform = BinarizingAnnotationTransform()
		
		self.transforms = [ {'src': f'ham_{idx}', 'dst': f'ham_html_{idx}', 'fn': BinarizingAnnotationTransform()} for idx in range(3)] + [
			{'src': 'ham_html_0', 'dst': 'text_tokens', 'fn': None},
			{'src': 'ham_html_0', 'dst': 'text_tokens', 'fn': None},
		]
	
	def setup(self, stage: str = None):
		dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data)
		
		# called on every GPU
		if stage == 'fit' or stage is None:
			self.train_set = SpacyPretokenizeYelpHat(split='train', **dataset_kwargs)
			self.val_set = SpacyPretokenizeYelpHat(split='val', **dataset_kwargs)
		
		if (stage == 'test' or stage is None) and not hasattr(self, 'test_set'):
			self.test_set = SpacyPretokenizeYelpHat(split='yelp50', **dataset_kwargs)
	
	def test_dataloader(self):
		return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate,
		                  num_workers=self.num_workers)


class YelpHat50DM(YelpHatDM):
	name = 'YelpHat-50'
	
	def __init__(self, **kwargs):
		super(YelpHat50DM, self).__init__(**kwargs)
	
	def setup(self, stage: str = None):
		dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data)
		
		# called on every GPU
		if stage == 'fit' or stage is None:
			self.train_set = SpacyPretokenizeYelpHat(split='train', **dataset_kwargs)
			self.val_set = SpacyPretokenizeYelpHat(split='val', **dataset_kwargs)
		
		if stage == 'test' or stage == 'predict' or stage is None:
			self.test_set = SpacyPretokenizeYelpHat(split='yelp50', **dataset_kwargs)
	
	def test_dataloader(self):
		return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
		
		
class YelpHat100DM(YelpHat50DM):

	name = 'YelpHat-100'
	
	def __init__(self, **kwargs):
		super(YelpHat100DM, self).__init__(**kwargs)
	
	def setup(self, stage: str = None):
		
		if stage == 'fit' or stage is None:
			super(YelpHat100DM, self).setup(stage)
		
		if (stage == 'test' or stage is None) and not hasattr(self, 'test_set'):
			self.test_set = SpacyPretokenizeYelpHat(split='yelp100', root=self.cache_path, n_data=self.n_data)
		
class YelpHat200DM(YelpHat50DM):
	
	name = 'YelpHat-200'
	
	def __init__(self, **kwargs):
		super(YelpHat200DM, self).__init__(**kwargs)
	
	def setup(self, stage: str = None):
		
		if stage == 'fit' or stage is None:
			super(YelpHat200DM, self).setup(stage)
		
		if (stage == 'test' or stage is None) and not hasattr(self, 'test_set'):
			self.test_set = SpacyPretokenizeYelpHat(split='yelp200', root=self.cache_path, n_data=self.n_data)
		
			
class CLSTokenYelpHatDM(YelpHatDM):
	
	def __init__(self, **kwargs):
		super(CLSTokenYelpHatDM, self).__init__(**kwargs)
	
	def prepare_data(self):
		super(CLSTokenYelpHatDM, self).prepare_data()
		
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
		b = super(CLSTokenYelpHatDM, self).collate(batch)
		
		# adding CLS token at the beginning
		bsz = b['token_ids'].size(0) # == batch_size or len(data) if len(data) < batch_size
		cls_ids = torch.tensor([self.vocab[SpecToken.CLS]]).repeat(bsz, 1)
		cls_pad = torch.tensor([0.]).repeat(bsz, 1)  # we contextualise the CLS token
		att_pad = torch.tensor([0.]).repeat(bsz, 1)
		
		# udpate classic batch with adding
		b.update({
			'token_ids': torch.cat((cls_ids, b['token_ids']), 1),
			'padding_mask': torch.cat((cls_pad, b['padding_mask']), 1),
			'a_true': torch.cat((att_pad, b['a_true']), 1),
		})
		return b
	
class CLSTokenYelpHat50DM(CLSTokenYelpHatDM):
	
	def __init__(self, **kwargs):
		super(CLSTokenYelpHat50DM, self).__init__(**kwargs)
	
	def setup(self, stage: str = None):
		dataset_kwargs = dict(root=self.cache_path, n_data=self.n_data)
		
		# called on every GPU
		if stage == 'fit' or stage is None:
			self.train_set = SpacyPretokenizeYelpHat(split='train', **dataset_kwargs)
			self.val_set = SpacyPretokenizeYelpHat(split='val', **dataset_kwargs)
		
		if (stage == 'test' or stage is None) and not hasattr(self, 'test_set'):
			self.test_set = SpacyPretokenizeYelpHat(split='yelp50', **dataset_kwargs)
	
	def test_dataloader(self):
		return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate, num_workers=self.num_workers)
	

class CLSTokenYelpHat100DM(CLSTokenYelpHat50DM):
	name = 'YelpHat-100'
	
	def __init__(self, **kwargs):
		super(CLSTokenYelpHat100DM, self).__init__(**kwargs)
	
	def setup(self, stage: str = None):
		
		if stage == 'fit' or stage is None:
			super(CLSTokenYelpHat100DM, self).setup(stage)

		if (stage == 'test' or stage is None) and not hasattr(self, 'test_set'):
			self.test_set = SpacyPretokenizeYelpHat(split='yelp100', root=self.cache_path, n_data=self.n_data)


class CLSTokenYelpHat200DM(CLSTokenYelpHat50DM):
	name = 'YelpHat-200'
	
	def __init__(self, **kwargs):
		super(CLSTokenYelpHat200DM, self).__init__(**kwargs)
	
	def setup(self, stage: str = None):
		
		if stage == 'fit' or stage is None:
			super(CLSTokenYelpHat200DM, self).setup(stage)
		
		if (stage == 'test' or stage is None) and not hasattr(self, 'test_set'):
			self.test_set = SpacyPretokenizeYelpHat(split='yelp200', root=self.cache_path, n_data=self.n_data)