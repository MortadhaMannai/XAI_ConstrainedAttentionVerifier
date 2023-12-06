import os

from os import path

import numpy as np
import pandas as pd

from data.functional import sample_subset
from data.yelp_hat.dataset import YelpHat, _EXTRACTED_FILES


class PretransformedYelpHat(YelpHat):

	
	def __init__(self, transformations : list[dict], prefix : str, split: str = 'yelp', root: str = path.join(os.getcwd(), '.cache'),  n_data: int = -1):
		super(PretransformedYelpHat, self).__init__(split=split, root=root)
		
		root = self.root(root)
		
		fname, fext = os.path.splitext(_EXTRACTED_FILES[split])
		if len(prefix) > 0 and prefix[0] != '.': prefix = '.' + prefix
		self.parquet_path = path.join(root, f'{fname}.pretransformed{prefix}{fext}')
		
		if path.exists(self.parquet_path):
			# load the cache file to data if file exist
			self.data = pd.read_parquet(self.parquet_path)
			# check
			for c in self.data.columns:
				if isinstance(self.data.loc[0, c], np.ndarray):
					self.data[c] = self.data[c].apply(lambda x: x.tolist())
		
		else:
		
			for transform in transformations:
				src = transform['src']
				dst = transform['dst']
				fn = transform['fn']
				
				input_args = self.data[src].to_dict('list')
				results = fn(**input_args)
				
				# control quality
				if isinstance(dst, list):
					if not isinstance(results, list or tuple):
						raise Exception('Resulted transformation not compatible with destined columns')
					if len(results) != len(dst):
						raise Exception(f'Incompatible result number: results = {len(results)} ; dst = {len(dst)}')
					
				self.data[dst] = results
		
		self.data = sample_subset(self.data, n_data, 'label')
