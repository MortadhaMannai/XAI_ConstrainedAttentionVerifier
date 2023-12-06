import os
import shutil

import pandas as pd
from os import path

from torchtext.data.datasets_utils import _create_dataset_directory
from torchtext.utils import download_from_url, extract_archive

from torch.utils.data import MapDataPipe

from data import ArgumentError
from modules.const import InputType
from modules.logger import log

DATASET_NAME = 'esnli'
NUM_CLASS = 3
INPUT = InputType.DUAL

URL = 'https://github.com/OanaMariaCamburu/e-SNLI/archive/refs/heads/master.zip'

_EXTRACTED_FILES = {
    'train': 'train.parquet',
    'val': 'dev.parquet',
    'test': 'test.parquet',
}

_MAP_FILE= {
	'train': ['esnli_train_1.csv', 'esnli_train_2.csv'],
	'val': ['esnli_dev.csv'],
	'test': ['esnli_test.csv']
}

ZIP_FILEPATH='_esnli.zip'

_LABEL_ITOS = ['neutral', 'entailment', 'contradiction']
_LABEL_STOI = {label: idx for idx, label in enumerate(_LABEL_ITOS)}

def download_format_dataset(root:str, split:str):
	"""
	Download and reformat dataset of eSNLI
	Args:
		root (str): cache folder where to find the dataset.
		split (str): among train, val, test
		n_data (int): maximum data to load. -1 to load entire data
	"""
	
	if path.basename(root) != DATASET_NAME:
		root = path.join(root, DATASET_NAME)
		# raise TypeError(f'Please concatenate root folder with dataset name: `{DATASET_NAME}`')
	
	# make a subdata set for dev purpose
	zip_path = path.join(root, ZIP_FILEPATH)
	parquet_path = path.join(root, _EXTRACTED_FILES[split])
	
	# download the zip file
	download_from_url(url=URL, path=zip_path)
	
	# decompress all files
	extract_archive(zip_path, root)
	
	# reformat csv files
	if not path.exists(parquet_path):
		original_file = [path.join(root, 'e-SNLI-master', 'dataset', f) for f in _MAP_FILE[split]]
		df = pd.concat([pd.read_csv(f) for f in original_file])
		df = _reformat_dataframe(df)
		df.to_parquet(parquet_path)
	
	return parquet_path
	
@_create_dataset_directory(dataset_name=DATASET_NAME)
def clean_up_dataset_cache(root):
	# clean up unnecessary files
	shutil.rmtree(path.join(root, 'e-SNLI-master'))

def _reformat_dataframe(data: pd.DataFrame):
		"""
		Remove unecessary columns, rename columns for better understanding. Notice that we also remove extra explanation
		columns.
		Args: data (pandas.DataFrame): Original data given by eSNLI dataset

		Returns:
			(pandas.DataFrame) clean data
		"""
		
		rename_cols = {
			'Sentence1': 'premise',
			'Sentence2': 'hypothesis',
			'gold_label': 'label',
			'Explanation_1': 'explanation',
			'Sentence1_marked_1': 'highlight_premise',
			'Sentence2_marked_1': 'highlight_hypothesis'
		}
		
		drop_cols = ['pairID', 'WorkerId'
		                       'Sentence1_Highlighted_1', 'Sentence2_Highlighted_1',
		             'Explanation_2', 'Sentence1_marked_2', 'Sentence2_marked_2',
		             'Sentence1_Highlighted_2', 'Sentence2_Highlighted_2',
		             'Explanation_3', 'Sentence1_marked_3', 'Sentence2_marked_3',
		             'Sentence1_Highlighted_3', 'Sentence2_Highlighted_3']
		
		if data.isnull().values.any():
			log.warning('Original dataset contain NA values, drop these lines.')
			data = data.dropna().reset_index()
		
		# rename column
		data = data.rename(
			columns=rename_cols
			# drop unneeded
		).drop(
			columns=drop_cols, errors='ignore'
		)[['premise', 'hypothesis', 'label', 'explanation', 'highlight_premise', 'highlight_hypothesis']]
		
		def correct_quote(txt, hl):
			"""
			Find the incoherent part in text and replace the corresponding in highlight part
			"""""
			
			# find different position between the 2
			diff = [i for i, (l, r) in enumerate(zip(txt, hl.replace('*', ''))) if l != r]
			
			# convert into list to be able to modify character
			txt, hl = list(txt), list(hl)
			
			idx = 0
			for pos_c, c in enumerate(hl):
				if c == '*': continue
				if idx in diff: hl[pos_c] = txt[idx]
				idx += 1
			
			hl = ''.join(hl)
			return hl
		
		# correct some error
		for side in ['premise', 'hypothesis']:
			data[side] = data[side].str.strip() \
				.str.replace('\\', '', regex=False) \
				.str.replace('*', '', regex=False)
			data[f'highlight_{side}'] = data[f'highlight_{side}'] \
				.str.strip() \
				.str.replace('\\', '', regex=False) \
				.str.replace('**', '*', regex=False)
			
			# replace all the simple quote (') by double quote (") as orignal phrases
			idx_incoherent = data[side] != data[f'highlight_{side}'].str.replace('*', '', regex=False)
			sub_data = data[idx_incoherent]
			replacement_hl = [correct_quote(txt, hl) for txt, hl in
			                  zip(sub_data[side].tolist(), sub_data[f'highlight_{side}'].tolist())]
			data.loc[idx_incoherent, f'highlight_{side}'] = replacement_hl
		
		return data

class ESNLI(MapDataPipe):
	
	NUM_CLASS = NUM_CLASS
	INPUT = INPUT
	LABEL_ITOS = _LABEL_ITOS
	
	def __init__(self, split: str = 'train', root: str = path.join(os.getcwd(), '.cache'), n_data: int = -1):
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
		if split not in _EXTRACTED_FILES.keys():
			raise ArgumentError(f'split argument {split} doesnt exist for eSNLI')
		
		root = self.root(root)
		self.split = split
		self.parquet_path = path.join(root, _EXTRACTED_FILES[split])
		self.zip_path = path.join(root, ZIP_FILEPATH)
		
		# download and prepare csv file if not exist
		download_format_dataset(root, split)
		
		# load the csv file to data
		self.data = pd.read_parquet(self.parquet_path)
		
		# if n_data activated, reduce the dataset equally for each class
		if n_data > 0:
			_unique_label = self.data['label'].unique()
			
			subset = [
				self.data[self.data['label'] == label]  # slice at each label
					.head(n_data // len(_unique_label)) # get the top n_data/3
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
	def root(cls, root): return path.join(root, DATASET_NAME)
	
	@classmethod
	def download_format_dataset(cls, root, split):
		return download_format_dataset(root, split)
	
	@classmethod
	def clean_cache(cls, root):
		return None
		
	
if __name__=='__main__':
	# Unit test
	
	from torch.utils.data import DataLoader
	
	cache_path = path.join(os.getcwd(), '..', '..', '..', '.cache', 'dataset')
	
	# To load the 3 at same time:
	# trainset, valset, testset = ESNLIDataPipe(root=cache_path)
	trainset = ESNLI(root=cache_path, split='train')
	
	train_loader = DataLoader(trainset, batch_size=3, shuffle=False)
	
	b = next(iter(train_loader))
	print('train batch:')
	print(b)
