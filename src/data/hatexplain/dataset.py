import os
import shutil

import pandas
import pandas as pd
from os import path

from datasets import load_dataset

from torch.utils.data import MapDataPipe

import numpy as np

from data import ArgumentError
from modules import map_np2list
from data.hatexplain.transforms import HeuristicTransform
from modules.const import InputType, INF
from modules.logger import log

DATASET_NAME = 'hatexplain'
INPUT = InputType.SINGLE

_EXTRACTED_FILES = {
	'train': 'train.parquet',
	'val': 'val.parquet',
	'test': 'test.parquet',
}

_HF_LABEL_ITOS = ['hatespeech', 'normal', 'offensive'] # huggingface's label
_LABEL_ITOS = ['normal', 'hatespeech', 'offensive'] # custom label


def download_format_dataset(root: str, split: str):
	"""Download and reformat dataset of eSNLI
	
	Parameters
	----------
	root : str
		cache folder where to find the dataset.
	split : str

	Returns
	-------
	str
		fpath to cache file
	"""
	
	if path.basename(root) != DATASET_NAME:
		root = path.join(root, DATASET_NAME)
	
	parquet_path = path.join(root, _EXTRACTED_FILES[split])
	if path.exists(parquet_path):
		return parquet_path
	
	huggingface_split = 'validation' if split == 'val' else split
	
	# download the dataset
	dataset = load_dataset(DATASET_NAME, split=huggingface_split, cache_dir=root)
	dataset = dataset.flatten()
	
	# Correct the example
	df = _reformat_dataframe(dataset.to_pandas(), split, cache=root)
	df.to_parquet(parquet_path)
	log.info(f'Process {split} and saved at {parquet_path}')
	
	return parquet_path
	

def clean_cache(root: str):
	shutil.rmtree(path.join(root, 'downloads'), ignore_errors=True)
	for fname in os.listdir(root):
		if fname.endswith('.lock'): os.remove(os.path.join(root, fname))
	

def _reformat_dataframe(data: pandas.DataFrame, split, cache):
	
	# Correct 1 example in train set
	if split == 'train':
		rationales = data.loc[1997, 'rationales']
		L = len(data.loc[1997, 'post_tokens'])
		rationales = [r[:L] for r in rationales]
		data.loc[1997, 'rationales'] = rationales
	
	data = data.apply(lambda column: [c.tolist() for c in column] if isinstance(column[0], np.ndarray) else column,axis=1)
	#data = map_np2list(data)
	
	# gold label = most voted label
	data['label'] = data['annotators.label'].apply(lambda x: np.bincount(x).argmax())
	
	# put back label into text
	data['label'] = data['label'].apply(lambda x: _HF_LABEL_ITOS[x]).astype('category') # decode from huggingface labels
	
	# rationale = average rationaled then binarize by 0.5 threshold
	data['rationale'] = data['rationales'].apply(lambda x: (np.mean([r.astype(float) for r in x], axis=0) >= 0.5).astype(int) if len(x) > 0 else x)
	
	# Case there exists empty rationale in positive label
	data['empty_rationale'] = data['rationale'].apply(lambda x: x.sum() == 0) & (data['label'] != 'normal')
	
	# else, pick the first one not empty
	def handle_empty_rationale(x):
		if x['label'] != 'normal' and x['rationale'].sum() == 0:
			for r in x['rationales']:
				if r.sum() > 0:
					x['rationale'] = r
					return x
		return x
	
	data = data.apply(handle_empty_rationale, axis=1)
	
	# if all rationales are zeros, change label into normal
	data['empty_rationale'] = data['rationale'].apply(lambda x: x.sum() == 0) & (data['label'] != 'normal')
	data.loc[data['empty_rationale'], 'label'] = 'normal'
	
	# make rationale for negative example, for padding coherent
	data['len_tokens'] = data.post_tokens.str.len()
	data['rationale'] = data.apply(lambda row: np.zeros(row['len_tokens'], dtype=np.int32) if len(row['rationale']) == 0 else row['rationale'], axis=1)
	
	# heuristic
	heuristic_transform = HeuristicTransform(
		batch_tokens=data['post_tokens'],
		batch_rationale=data['rationale'],
		cache=cache,
		mask_value=0.,
		pos_filter=['ADJ']
	)
	heuristics = heuristic_transform(data['post_tokens'].tolist())
	
	data['heuristic'] = pd.Series(heuristics)
	
	data = data.drop(columns=['len_tokens', 'empty_rationale', 'rationales', 'annotators.label', 'annotators.target'])
	
	# put back into list
	data = map_np2list(data)
	
	return data


class HateXPlain(MapDataPipe):
	
	INPUT = INPUT
	LABEL_ITOS = _LABEL_ITOS
	
	def __init__(self, split: str = 'train', root: str = path.join(os.getcwd(), '.cache'), n_data: int = -1):
		
		# assert
		if split not in _EXTRACTED_FILES.keys():
			raise ArgumentError(f'split argument {split} doesnt exist for {type(self).__name__}')
		
		root = self.root(root)
		self.split = split
		
		# download and prepare parquet file if not exist
		self.parquet_path = download_format_dataset(root=root, split=split)
		
		# load the parquet file to data
		self.data = pd.read_parquet(self.parquet_path)
		
		self.data = map_np2list(self.data)
		
		# if n_data activated, reduce the dataset equally for each class
		if n_data > 0:
			_unique_label = self.data['label'].unique()
			
			subsets = [
				self.data[self.data['label'] == label]  # slice at each label
					.head(n_data // len(_unique_label))  # get the top n_data/3
				for label in _unique_label
			]
			self.data = pd.concat(subsets).reset_index(drop=True)
	
	def __getitem__(self, index: int):
		
		# Load data and get label
		if index >= len(self): raise IndexError  # meet the end of dataset
		
		sample = self.data.loc[index].to_dict()
		return sample
	
	def __len__(self):
		"""
		Denotes the total number of samples
		Returns: int
		"""
		return len(self.data)
	
	@classmethod
	def root(cls, root):
		return path.join(root, DATASET_NAME)
	
	@classmethod
	def download_format_dataset(cls, root, split):
		return download_format_dataset(root, split)
	
	@classmethod
	def clean_cache(cls, root):
		return clean_cache(root)

if __name__ == '__main__':
	# Unit test
	
	from torch.utils.data import DataLoader
	
	cache_path = path.join(os.getcwd(), '.cache', 'dataset')
	
	# To load the 3 at same time:
	# trainset, valset, testset = ESNLIDataPipe(root=cache_path)
	trainset, valset = HateXPlain(root=cache_path, split=('train', 'val'))
	
	train_loader = DataLoader(trainset, batch_size=3, shuffle=False)
	for b in train_loader:
		print('train batch:')
		print(b)
		break
	
	val_loader = DataLoader(valset, batch_size=1, shuffle=False)
	for b in train_loader:
		print('val batch:')
		print(b)
		break
