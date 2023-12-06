import os

import numpy as np
import pandas as pd
from os import path

from tqdm import trange

from data.esnli.dataset import ESNLI, _EXTRACTED_FILES
from modules.logger import log

_SEGMENT_SIZE = 10000

class PretransformedESNLI(ESNLI):
	
	def __init__(self, transformations: dict, column_name: dict, prefix:str='',  split: str = 'train', root: str = path.join(os.getcwd(), '.cache'), n_data: int = -1):
		"""

		Args:
			split       (str):
			cache_path  (str):
			n           (int): max of data to be loaded
			shuffle     (bool): shuffle if load limited data
						If n is precised and shuffle = True, dataset will sample n datas.
						If n is precised and shuffle = False, dataset will take only n first datas.
		"""
		
		# assert
		super(PretransformedESNLI, self).__init__(split=split, root=root)
		assert transformations.keys() == column_name.keys(), f'Incoherent item between transforms and column_name. \n\tTransforms: {list(transformations.keys())} \n\tColumns: {list(column_name.keys())}'
		
		root = self.root(root)
		
		fname, fext = os.path.splitext(_EXTRACTED_FILES[split])
		if len(prefix) > 0 and prefix[0] != '.': prefix = '.' + prefix
		self.parquet_path = path.join(root, f'{fname}.pretransformed{prefix}{fext}')
		
		if path.exists(self.parquet_path):
			# load the cache file to data if file exist
			self.data = pd.read_parquet(self.parquet_path)
			# check
			for c in self.data.columns:
				if isinstance(self.data.loc[0,c], np.ndarray):
					self.data[c] = self.data[c].apply(lambda x: x.tolist())
				
		else:
			segment_path = path.join(root, 'segments')
			n_data = len(self.data)
			n_segments = (n_data // _SEGMENT_SIZE) + int(n_data % _SEGMENT_SIZE > 0)
			log.info(f'Split {split} has {n_segments} segments x {_SEGMENT_SIZE} rows/segment')
			
			# if not exists segment of dataset: split into segments and save in "segments"
			os.makedirs(segment_path, exist_ok=True)
			segment_files = [fname for fname in os.listdir(segment_path) if split in fname]
			
			# segment files if not yet done
			if len(segment_files) == 0:
				for idx in range(n_segments):
					offset = idx * _SEGMENT_SIZE
					self.data[offset:offset+_SEGMENT_SIZE].to_parquet(path.join(segment_path, f'{fname}.{idx}{fext}'), index=False)
				log.info(f'Segments saved at {segment_path}.')
			
			# pick up the last one treated to continue
			pretransformed_segments = [fname for fname in segment_files if 'pretransformed' in fname]
			starting_idx = len(pretransformed_segments)
			
			heuristic_col = column_name['heuristic']
			
			for idx in trange(starting_idx, n_segments, desc=f'{split} segments'):
				data = pd.read_parquet(path.join(segment_path, f'{fname}.{idx}{fext}'))
				# pretransform if not exist, then save to cache
				
				for column, transform in transformations.items():
					if column == 'heuristic':
						continue
					new_column = column_name[column]
					data[new_column] = pd.Series(transform(data[column]))
				
				heuristic_transform = transformations['heuristic']
				result = heuristic_transform(premise=data['premise'], hypothesis=data['hypothesis'])
				
				for side in ['premise', 'hypothesis']:
					score = result[side]
					mask = result[f'{side}_mask']
					# get length (# of token) for each sentence
					sent_length = mask.sum(dim=1).tolist()
					# flatten the score by masking padding scores, then resplit by sentence length
					unpad_heuristic = score[mask.bool()].split(sent_length)
					# transform into list to save into cache
					unpad_heuristic = [h.tolist() for h in unpad_heuristic]
					data[f'{side}_{heuristic_col}'] = pd.Series(unpad_heuristic)
				
				data.to_parquet(path.join(segment_path, f'{fname}.pretransformed.{idx}{fext}'), index=False)
			
			# fusion all the pretransformed segments
			log.info(f'Concatenating {split} segments')
			segment_df = [pd.read_parquet(path.join(segment_path, f'{fname}.pretransformed.{idx}{fext}')) for idx in range(n_segments)]
			self.data = pd.concat(segment_df)
			self.data.to_parquet(self.parquet_path, index=False)
			log.info(f'Save pretransform {split} eSNLI at: {self.parquet_path}')
			# clean the intermediates
			
		# if n_data activated, reduce the dataset equally for each class
		if n_data > 0:
			_unique_label = self.data['label'].unique()
			
			subset = [
				self.data[self.data['label'] == label]  # slice at each label
					.head(n_data // len(_unique_label)) # get the top n_data/3
				for label in _unique_label
			]
			self.data = pd.concat(subset).reset_index(drop=True)
	
		
	
if __name__=='__main__':
	# Unit test
	
	from torch.utils.data import DataLoader
	
	cache_path = path.join(os.getcwd(), '..', '..', '..', '.cache', 'dataset')
	
	# To load the 3 at same time:
	# trainset, valset, testset = ESNLIDataPipe(root=cache_path)
	trainset = PretransformedESNLI(root=cache_path, split='train')
	
	train_loader = DataLoader(trainset, batch_size=3, shuffle=False)
	
	b = next(iter(train_loader))
	print('train batch:')
	print(b)
