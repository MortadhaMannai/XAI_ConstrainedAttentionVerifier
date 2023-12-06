import json
import os
from os import path
from typing import List, Union

import spacy
from spacy.tokens import Doc
from torch.nn import Module

from modules import log
from modules.const import INF

DEFAULT_POS_FILTER = ['ADJ', 'VERB', 'NOUN']

class HeuristicTransform(Module):
	
	def __init__(self,
	             batch_tokens: List[List[str]]=None,
	             batch_rationale: List[List[Union[int, bool]]]=None,
	             mask_value:float=0,
	             cache:str=None,
	             spacy_model=spacy.load('en_core_web_sm'),
	             pos_filter: List[str]=None):
		"""Construct heuristic map from dataset based on frequent annotated tokens and POS filter
		
		Parameters
		----------
		batch_tokens : list
			tokens from train set
		batch_rationale : list
			binary map. have to match with batch tokens
		cache : str, optional, default None
			fpath to store caching files for this function
		spacy_model : spacy.Tokenizer, optional
			spacy tokenizer
		pos_filter : list, optional
		"""
		
		super(HeuristicTransform, self).__init__()
		self.spacy_model = spacy_model
		self.POS_FILTER = pos_filter if pos_filter is not None else DEFAULT_POS_FILTER
		self.cache = cache
		self.freq_path = path.join(self.cache if cache is not None else os.getcwd(), 'annotation_lexical_frequency.json')
		self.token_freq = self._token_frequency(batch_tokens, batch_rationale)
		self.MASK_VAL = mask_value
		
	def _token_frequency(self, batch_tokens: List[List[str]], batch_rationale: List[List[Union[int, bool]]]):
		"""Compute annotation frequency for each token. If `self.freq_path` exists, the function reuses this caching file
		
		Parameters
		----------
		batch_tokens : list of str
			list of token sequence in train set
		batch_rationale : list of binary int or list of boolean
			list of rational (binary) sequence in train set

		Returns
		-------
		token_freq: annotation frequency for each token in train set
		"""
		
		if self.cache is not None and path.exists(self.freq_path):
			with open(self.freq_path, 'r') as f:
				token_freq = json.load(f)
				log.info(f'Load annotation frequency from {self.freq_path}')
			return token_freq
		
		if batch_tokens is None or batch_rationale is None:
			raise ValueError(f'The caching token frequency is not established, please feed tokens and rationales')
		
		token_freq = dict()
		
		flatten_token = [tk for sent in batch_tokens for tk in sent]
		flatten_rationale = [r for sent in batch_rationale for r in sent]
		
		for t, r in zip(flatten_token, flatten_rationale):
			if r: token_freq[t] = token_freq.get(t, 0) + 1
		
		total_freq = sum(token_freq.values())
		token_freq = {k: v / total_freq for k, v in token_freq.items()}
		token_freq = dict(sorted(token_freq.items(), key=lambda item: -item[1]))
		
		if self.cache is not None:
			with open(self.freq_path, 'w') as f:
				json.dump(token_freq, f, indent='\t')
				log.info(f'Save annotation frequency in {self.freq_path}')
		
		return token_freq
	
	
	def forward(self, batch_tokens: List[List[str]]):
		"""Convert each token sequence into a heuristic sequence. Heuristic values are token frequency. Ignored tokens are valued -const.INF.
		
		Parameters
		----------
		batch_tokens : list of string sequences
			batch of token sequence

		Returns
		-------
		heuristic : list of float
			corresponding heuristic values
		"""
		docs = [Doc(self.spacy_model.vocab, words=sent) for sent in batch_tokens]
		docs = list(self.spacy_model.pipe(docs))
		pos_mask = [[tk.pos_ in self.POS_FILTER for tk in d] for d in docs]
		stop_mask = [[not tk.is_stop for tk in d] for d in docs]
		mask = [[p and s for p, s in zip(pos_, stop_)] for pos_, stop_ in zip(pos_mask, stop_mask) ]
		
		## Count words
		heuristics = []
		for sent_tokens, sent_mask in zip(batch_tokens, mask):
			heuris_map = [self.token_freq.get(tk, 0) for tk in sent_tokens]
			heuris_map = [h if m else self.MASK_VAL for h, m in zip(heuris_map, sent_mask) ]
			heuristics.append(heuris_map)
			
		return heuristics

		