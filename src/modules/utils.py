import json
import os
from os import path
import shutil
from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd
import torch

from modules.const import INF
from modules.logger import log

def rescale(attention: torch.Tensor, mask: torch.Tensor):
	"""
	Project min value to 0 and max value to 1.
	
	:param attention:
	:type attention:
	:param mask:
	:type mask:
	:return:
	:rtype:
	"""
	v_max = torch.max(attention + mask.float() * -INF, dim=1, keepdim=True).values
	v_min = torch.min(attention + mask.float() * INF, dim=1, keepdim=True).values
	v_min[v_min == v_max] = 0.
	rescale_attention = (attention - v_min)/(v_max - v_min)
	rescale_attention[mask] = 0.
	return rescale_attention


def hex2rgb(hex):
	rgb = [int(hex[i:i + 2], 16) for i in [1,3,5] ]
	return rgb


def hightlight(words: List[str], weights: Union[np.ndarray, torch.tensor, list], color:Union[str, Tuple[int]]=None):
	"""Build HTML that highlights words based on its weights
	
	Parameters
	----------
	words : list of token (str)
		1-D list iterable, containing tokens
	weights : numpy.ndarray, torch.tensor or list
		weight along text
	color : str or tuple, optional
		highlight color, in hexadecimal (ex: `#FF00FF`) or rgb (ex: `(11,12,15)`)
		
	Returns
	-------
	str
	
	Examples
	-------
	```python
		from IPython.core.display import display, HTML
		highlighted_text = hightlight_txt(lemma1[0], a1v2)
		display(HTML(highlighted_text))
		```
	"""
	MAX_ALPHA = 0.8
	
	if isinstance(weights, np.ndarray):
		weights = torch.from_numpy(weights)
	elif isinstance(weights, list):
		weights = torch.tensor(weights)
	
	weights = weights.float()
		
	w_min, w_max = torch.min(weights), torch.max(weights)
	
	w_norm = (weights - w_min)/((w_max - w_min) + (w_max == w_min)*w_max)
	
	# make color
	# change to rgb if given color is hex
	if color is None:
		color = [135, 206, 250]
	elif color[0] == '#' and len(color) == 7:
		color = hex2rgb(color)
	w_norm = (w_norm / MAX_ALPHA).tolist()

	# wrap each token in a span
	highlighted_words = [f'<span style="background-color:rgba{(*color, w)};">' + word + '</span>' for word, w in zip(words, w_norm)]

	# concatenate spans into a string
	return ' '.join(highlighted_words)

def report_score(scores: dict, logger, score_dir=None) -> None:
	"""
	Report scores into score.json and logger
	:param scores: dictionary that has reported scores
	:type scores: dict
	:param logger: Tensorboard logger. Report into hyperparameters
	:type logger: TensorBoardLogger
	:param score_dir: directory to find score.json
	:type score_dir: str
	:return: None
	:rtype: None
	"""
	
	# remove 'TEST/' from score dicts:
	scores = [{k.replace('TEST/', ''): v for k, v in s.items()} for s in scores]
	
	for idx, score in enumerate(scores):
		log.info(score)
		logger.log_metrics(score)
		
		if score_dir is not None:
			os.makedirs(score_dir, exist_ok=True)
			src = path.join(logger.log_dir, 'hparams.yaml')
			dst = path.join(score_dir, 'hparams.yaml')
			shutil.copy2(src, dst)
			
		score_path = path.join(score_dir or logger.log_dir, f'score{"" if idx == 0 else "_" + str(idx)}.json')
		
		with open(score_path, 'w') as fp:
			json.dump(score, fp, indent='\t')
			log.info(f'Score is saved at {score_path}')
		
def flatten_dict(nested: dict, sep:str='.') -> dict:
	"""
	Convert a nested dictionary into a flatt dictionary
	
	Parameters
	----------
	nested : dict
		nested dictionary to be flattened
	sep : str
		separator between parent key and child key

	Returns
	-------
	dict
		flat dictionary

	"""
	assert isinstance(nested, dict), f"Only flatten dictionary, nested given is of type {type(nested)}"
	
	flat = dict()
	
	for current_key, current_value in nested.items():
		
		# if value is a dictionary, then concatenate its key with current key
		if isinstance(current_value, dict):
			flat_item = flatten_dict(current_value, sep=sep)
			
			flat.extend()
			
			for child_key, child_value in flat_item.items():
				flat[current_key + sep + child_key] = child_value
				
		else:
			flat[current_key] = current_value
			
	return flat
	
def quick_flatten_dict(nested: dict, sep:str= '.') -> dict:
	"""New version of how to flat a dictionary using pandas
	
	Parameters
	----------
	nested : dict
		nested dictionary to be flattened
	sep : str
		separator between parent key and child key

	Returns
	-------
	dict
		flat dictionary

	"""
	return pd.json_normalize(nested, sep=sep).to_dict(orient='records')[0]

def map_list2dict(batch: Union[List[Dict],Dict]) -> dict:
	"""convert list of dict to dict of list
	
	Parameters
	----------
	batch : List[Dict] or Dict
		batch of dictionaries
		
	Returns
	-------
	dict
		dictionary where data are batched in each key.
	"""
	if isinstance(batch, dict):
		return {k: list(v) for k, v in batch.items()}  # handle case where no batch
	return {k: [row[k] for row in batch] for k in batch[0]}

def recursive_list2dict(batch: Union[List[Dict], Dict]):
	
	if isinstance(batch, list):
		
		if isinstance(batch[0], dict):
			batch = {k: [row[k] for row in batch] for k in batch[0]}
		
		elif isinstance(batch[0], list):
			batch =  [item for sub_list in batch for item in sub_list]
	
	if isinstance(batch, dict):
		
		for k in batch:
			# flatten all list of dict
			batch[k] = recursive_list2dict(batch[k])
	
	return batch

def map_np2list(df: pd.DataFrame):
	"""Auto convert numpy columns into list columns
    
    Parameters
    ----------
    df : pd.DataFrame
        entire data

    Returns
    -------
    df : pd.DataFrame
        formatted data
    """
	
	return df.apply(lambda column: [c.tolist() for c in column] if isinstance(column[0], np.ndarray) else column)