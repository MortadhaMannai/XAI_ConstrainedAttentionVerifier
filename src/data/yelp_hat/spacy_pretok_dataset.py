import os
import shutil

import pandas as pd
from os import path

import spacy

import numpy as np

from data.hatexplain.transforms import HeuristicTransform
from data.yelp_hat.dataset import YelpHat, DATASET_NAME
from data.yelp_hat.utils import yelp_hat_ham, yelp_hat_token
from modules import INF, map_np2list
from modules.logger import log
	
def _reformat_dataframe(data: pd.DataFrame, spacy_model=None, lemma:bool=True, lower:bool=True, cache=os.getcwd()):
	
	if spacy_model is None: spacy_model=spacy.load('en_core_web_sm')
	
	# Binarizing human attention map
	for idx in range(3):
		data[f'ham_{idx}'] = data[f'ham_html_{idx}'].apply(lambda x: yelp_hat_ham(x, spacy_model)).apply(lambda x: np.array(x))
		
	# Pre tokenize
	data['text_tokens'] = data['ham_html_0'].apply(lambda x: yelp_hat_token(x, spacy_model, lemma, lower))
	
	# Drop incoherent attention maps samples
	data_drop = data[(data['ham_0'].str.len() == data['ham_1'].str.len()) & (data['ham_1'].str.len() == data['ham_2'].str.len())].reset_index(drop=True)
	n_drop = len(data) - len(data_drop)
	
	if n_drop > 0:
		log.warning(f'Drop {n_drop} samples because HAMs are not compatibles')
		data = data_drop
	
	# Synthetize the rationale
	data['ham'] = data.apply(lambda row: ((row['ham_0'] + row['ham_1'] + row['ham_2']) / 3 >= 0.5).astype(int), axis=1)
	data['cam'] = data.apply(lambda row: row['ham_0'] * row['ham_1'] * row['ham_2'], axis=1)
	data['sam'] = data.apply(lambda row: ((row['ham_0'] + row['ham_1'] + row['ham_2']) > 0).astype(int), axis=1)
	
	# convert numpy into list:
	data = map_np2list(data)
	
	if (data.text_tokens.str.len() != data.ham_0.str.len()).any():
		mismatch_index = data.index[data.text_tokens.str.len() != data.ham_0.str.len()]
		raise ValueError(f'Tokens and Rationale dimension mismatch at: {mismatch_index}')
		
	# heuristic
	heuristic_transform = HeuristicTransform(
		batch_tokens=data['text_tokens'],
		batch_rationale=data['ham'],
		spacy_model=spacy_model,
		mask_value=0,
		cache=cache)
	
	heuristics = heuristic_transform(data['text_tokens'].tolist())
	data['heuristic'] = pd.Series(heuristics)
	
	return data

class SpacyPretokenizeYelpHat(YelpHat):
	
	def __init__(self, split: str = 'yelp', root: str = path.join(os.getcwd(), '.cache'), n_data: int = -1, spacy_model=None, lemma:bool=True, lower:bool=True):
		
		super(SpacyPretokenizeYelpHat, self).__init__(split=split, root=root, n_data=n_data)
		root = self.root(root)
		
		fname, fext = os.path.splitext(self.parquet_path)
		fprep = 'pretokenized' + ('_lower' if lower else '') + ('_lemma' if lemma else '')
		self.parquet_path = path.join(root, f'{fname}.{fprep}{fext}')
		
		if path.exists(self.parquet_path):
			self.data = pd.read_parquet(self.parquet_path)
			self.data = map_np2list(self.data)
		else:
			self.data = _reformat_dataframe(self.data, spacy_model, lemma, lower, cache=root)
			self.data.to_parquet(self.parquet_path)
			log.info(f'Save yelp subset {split} at: {self.parquet_path}')
		
		# if n_data activated, reduce the dataset equally for each class
		if n_data > 0:
			_unique_label = self.data['label'].unique()
			
			subset = [
				self.data[self.data['label'] == label]  # slice at each label
					.head(n_data // len(_unique_label))  # get the top n_data/3
				for label in _unique_label
			]
			self.data = pd.concat(subset).reset_index(drop=True)
	
	@classmethod
	def root(cls, root):
		return path.join(root, DATASET_NAME)


if __name__ == '__main__':
	# Unit test
	
	from torch.utils.data import DataLoader
	
	cache_path = path.join(os.getcwd(), '..', '..', '..', '.cache', 'dataset')
	
	trainset = SpacyPretokenizeYelpHat(root=cache_path, split='train')
	
	train_loader = DataLoader(trainset, batch_size=3, shuffle=False)
	b = next(iter(train_loader))
	print('train batch:')
	print(b)
