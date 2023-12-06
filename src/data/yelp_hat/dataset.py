import os
import shutil

import pandas as pd
from os import path

from sklearn.model_selection import train_test_split

from torch.utils.data import MapDataPipe

import numpy as np
from torchtext.utils import download_from_url, extract_archive

from data import ArgumentError
from modules.const import InputType
from modules.logger import log

DATASET_NAME = 'yelp-hat'
INPUT = InputType.SINGLE

_EXTRACTED_FILES = {
	'yelp': 'yelp.parquet',
	'yelp50': 'yelp50.parquet',
	'yelp100': 'yelp100.parquet',
	'yelp200': 'yelp200.parquet',
	
	'train': 'train.parquet',
	'val': 'val.parquet',
}

_SUBSET = {
	'ham_part1(50words).csv': 'yelp50.parquet',
	'ham_part6(100words).csv': 'yelp100.parquet',
	'ham_part8(200words).csv': 'yelp200.parquet'
}

_VAL_SPLIT = 0.3

URL = 'https://github.com/cansusen/Human-Attention-for-Text-Classification/archive/205c1552bc7be7ec48623d79d85d4c6fbfe62362.zip'

_LABEL_ITOS = ['negative', 'postive']  # https://github.com/cansusen/Human-Attention-for-Text-Classification

def download_format_dataset(root: str, split: str='yelp'):
	"""
	Download and reformat dataset of eSNLI
	Args:
		root (str): cache folder where to find the dataset.
		split (str): among train, val, test
		n_data (int): maximum data to load. -1 to load entire data
	"""
	
	if path.basename(root) != DATASET_NAME:
		root = path.join(root, DATASET_NAME)
	
	zip_path = download_from_url(URL, root=root, path=path.join(root, f'{DATASET_NAME}.zip'))
	extracted_path = path.join(root, 'caching')
	files = extract_archive(from_path=zip_path, to_path=extracted_path)
	files = [f for f in files if f.endswith('.csv')]
	
	# If fpath exists already, ignore doing things
	parquet_path = path.join(root, _EXTRACTED_FILES[split])
	if path.exists(parquet_path): return parquet_path
	
	# Place .csv files at the level of dataset
	for f in files: shutil.copy2(f, extracted_path)
	
	# Special case of part7.csv: contains 2 HAMs and 4 HAMs for some sentences
	df = pd.read_csv(path.join(extracted_path, 'ham_part7.csv'))
	
	# 1. Duplicate of 2: drop all
	duplicates = df.groupby(df['Input.text'].tolist(), as_index=False).size()
	dupli_2 = duplicates[duplicates['size'] < 3] # finds dupli_2
	df = df[~df['Input.text'].isin(dupli_2['index'])]
	
	# 2. Duplicate of 4:
	df = df.groupby('Input.text').head(3).reset_index(drop=True)
	
	# Check if no other duplicate in part7
	duplicates = df.groupby(df['Input.text'].tolist(),as_index=False).size()
	duplicated_values = duplicates['size'].unique()
	if len(duplicated_values) == 1 and duplicated_values[0] == 3:
		log.debug('Correctly handle part7.csv')
	else:
		log.error(f'Unsuccessfully handle part7.csv. Duplicated HAM: {duplicated_values}')
		raise ArithmeticError
	
	df.to_csv(path.join(extracted_path, 'ham_part7.csv'), index=False)
	
	# Special case of part5.csv: drop duplicate 2
	df = pd.read_csv(path.join(extracted_path, 'ham_part5.csv'))
	duplicates = df.groupby(df['Input.text'].tolist(), as_index=False).size()
	dupli_2 = duplicates[duplicates['size'] == 2]  # finds dupli_2
	df = df[~df['Input.text'].isin(dupli_2['index'])]
	df.to_csv(path.join(extracted_path, 'ham_part5.csv'), index=False)
	
	# Now reformat every files and save them to extracted
	training_sets = []
	files = [path.join(extracted_path, f) for f in os.listdir(extracted_path) if f.endswith('.csv')]

	for f in files:
		df = pd.read_csv(f).drop(columns=['index'], errors='ignore')
		df = _reformat_dataframe(df)
		
		parquet_path = _SUBSET.get(path.basename(f), False)
		
		if parquet_path:
			parquet_path = path.join(root, parquet_path)
			df.to_parquet(parquet_path)
			log.info(f'Save yelp subset at: {parquet_path}')
		else:
			training_sets.append(df)
	
	training_df = pd.concat(training_sets, ignore_index=True)
	training_df.to_parquet(path.join(root, _EXTRACTED_FILES['yelp']))
	log.info(f'Save clean dataset at {path.join(root, _EXTRACTED_FILES["yelp"])}')
	
	# Spliting data set into train and val
	train_df, val_df = train_test_split(training_df, test_size=_VAL_SPLIT)
	train_df = train_df.reset_index(drop=True)
	val_df = val_df.reset_index(drop=True)
	train_df.to_parquet(path.join(root, _EXTRACTED_FILES['train']))
	log.info(f'Save training set at {path.join(root, _EXTRACTED_FILES["train"])}')
	val_df.to_parquet(path.join(root, _EXTRACTED_FILES['val']))
	log.info(f'Save validation dataset at {path.join(root, _EXTRACTED_FILES["val"])}')
	
	return path.join(root, _EXTRACTED_FILES[split])
	

def clean_cache(root: str):
	shutil.rmtree(path.join(root, 'caching'), ignore_errors=True)
		
def _reformat_dataframe(data: pd.DataFrame):
	
	# Rename column
	data = data.rename(columns={
		'Answer.Q1Answer': 'human_label_',
	    'Input.text': 'text',
	    'Input.label': 'label',
		'Answer.html_output': 'ham_html_'
	})
	
	# Sliding into 3 subset to get 3HAMs
	dfs = [data.loc[0::3, ['text', 'label']].reset_index(drop=True)]
	for idx in range(3):
		_data = data.loc[idx::3, ['ham_html_', 'human_label_']]
		_data = _data.reset_index(drop=True).add_suffix(str(idx))
		dfs += [_data]
		
	data = pd.concat(dfs, axis=1)
	
	return data

class YelpHat(MapDataPipe):
	
	INPUT=INPUT
	LABEL_ITOS=_LABEL_ITOS
	
	def __init__(self, split: str = 'yelp', root: str = path.join(os.getcwd(), '.cache'),  n_data: int = -1):
		"""
		
		Parameters
		----------
		split :
		root :
		n_data :
		
		See Also
		----------
			https://github.com/cansusen/Human-Attention-for-Text-Classification
		
		"""
		
		# assert
		if split not in _EXTRACTED_FILES.keys():
			raise ArgumentError(f'split argument {split} doesnt exist for {type(self).__name__}')
		
		root = self.root(root)
		self.split = split
		
		# download and prepare csv file if not exist
		self.parquet_path = download_format_dataset(root=root, split=split)
		
		# load the csv file to data
		self.data = pd.read_parquet(self.parquet_path)
		log.info(f'Load dataset from {self.parquet_path}')
		
		# if n_data activated, reduce the dataset equally for each class
		if n_data > 0:
			_unique_label = self.data['label'].unique()
			
			subset = [
				self.data[self.data['label'] == label]  # slice at each label
					.head(n_data // len(_unique_label))  # get the top n_data/3
				for label in _unique_label
			]
			self.data = pd.concat(subset).reset_index(drop=True)
	
	def __getitem__(self, index: int):
		"""

		Args:
			index ():

		Returns:

		"""
		
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
	def download_format_dataset(cls, root, split='yelp'):
		return download_format_dataset(root, split)
	
	@classmethod
	def clean_cache(cls, root):
		return clean_cache(root)


if __name__ == '__main__':
	# Unit test
	
	from torch.utils.data import DataLoader
	
	cache_path = path.join(os.getcwd(), '..', '..', '..', '.cache', 'dataset')
	
	trainset = YelpHat(root=cache_path, split='train')
	
	train_loader = DataLoader(trainset, batch_size=3, shuffle=False)
	b = next(iter(train_loader))
	print('train batch:')
	print(b)
