import string

import spacy
import torch
from typing import Union, List, Mapping
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab.vectors import pretrained_aliases as pretrained, Vectors

from modules.const import EPS, INF


class GoldLabelTransform(Module):
	
	
	LABEL_MAPS = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

	def __init__(self, label_map:Mapping[str, int] = None):
		"""Turn eSNLI gold label into corresponding class id
		
		Parameters
		----------
		label_map : dict
			map label string into integer.
		"""
		super(GoldLabelTransform, self).__init__()
		self.lmap_ = label_map if label_map is not None else self.LABEL_MAPS
	
	def forward(self, labels: Union[list, str]):
		if isinstance(labels, str):
			return self.lmap_[labels]
		return [self.lmap_[l] for l in labels]


class HighlightTransform(Module):
	
	def __init__(self):
		"""Turn human highlights into boolean mask, True if it's attention
		"""
		super(HighlightTransform, self).__init__()
	
	def forward(self, highlight: Union[list, str]):
		masks = [list()] * len(highlight)
		for idx, phrase in enumerate(highlight):
			mask = []
			is_highlight = False
			for token in phrase:
				if token == '*':
					is_highlight = not is_highlight
					continue
				mask.append(is_highlight and token not in string.punctuation)
			masks[idx] = mask
		return masks


class HeuristicTransform(Module):
	
	def __init__(self,
	             vectors: str or Vectors,
	             spacy_model:spacy.Language=spacy.load('en_core_web_sm'),
	             cache:str=None):
		"""Construct heuristic map from dataset based on word pair similarities and POS filter
		
		Parameters
		----------
		vectors : string for Vectors
			any pretrained vector to compute similarity
		spacy_model : spacy.Language
		cache : str
			fpath to store caching files for this function
		"""
		super(HeuristicTransform, self).__init__()
		self.sm = spacy_model
		if isinstance(vectors, str):
			vectors = pretrained[vectors](cache=cache)
		self.vectors = vectors
		self.POS_FILTER = ['VERB', 'NOUN', 'ADJ']
	
	def forward(self, premise: List[List[str]], hypothesis: List[List[str]]):
		"""Convert each token sequence into a heuristic sequence. Heuristic values are token frequency. Ignored tokens are valued -const.INF.

		Parameters
		----------
		premise : list of string sequences
			batch of token sequence
		premise : list of string sequences
			batch of token sequence

		Returns
		-------
		heuristic : dict {'premise': [], 'hypothesis': []}
			corresponding heuristic values
		"""
		batch = {'premise': premise, 'hypothesis': hypothesis}
		vectors = {}
		mask = {}
		heuristic = {}
		padding_mask = {}
		
		for side, texts in batch.items():
			docs = list(self.sm.pipe(texts))
			
			padding = [torch.tensor([1] * len(d)) for d in docs]
			
			# POS-tag mask: True on informative tokens
			# pos = [[tk.pos_ for tk in d] for d in docs]
			pos_mask = [torch.tensor([(not tk.is_stop) and (tk.pos_ in self.POS_FILTER) for tk in d]) for d in docs]
			
			for idx in range(len(pos_mask)):
				if (~pos_mask[idx]).all():
					pos_mask[idx] = torch.tensor([not tk.is_stop for tk in docs[idx]])
				if (~pos_mask[idx]).all():
					pos_mask[idx] = torch.tensor([True for _ in docs[idx]])  # Uniform attention
			
			pos_mask = pad_sequence(pos_mask, batch_first=True, padding_value=False)
			padding = pad_sequence(padding, batch_first=True, padding_value=0)
			
			tokens = [[tk.lemma_.lower() for tk in d] for d in docs]
			v = [self.vectors.get_vecs_by_tokens(tk) for tk in tokens]
			v = pad_sequence(v, batch_first=True)
			v_norm = v / (v.norm(dim=2)[:, :, None] + EPS)
			
			vectors[side] = v_norm
			mask[side] = pos_mask
			padding_mask[side] = padding
		
		# similarity matrix
		similarity = torch.bmm(vectors['premise'], vectors['hypothesis'].transpose(1, 2))
		
		# apply mask
		for side, dim in zip(batch.keys(), [2, 1]):
			heuristic[side] = similarity.sum(dim).masked_fill_(~mask[side], - INF)
	
		for side in batch.keys():
			heuristic[f'{side}_mask'] = padding_mask[side]
			
		return heuristic
	