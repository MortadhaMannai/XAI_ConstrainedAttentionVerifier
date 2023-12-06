from typing import Union


def mean(x: Union[dict, list]):
	"""
	Average values in x array
	:param x:
	:type x:
	:return:
	:rtype:
	"""
	if isinstance(x, dict):
		return sum(x.values()) / len(x)
	if isinstance(x, list):
		return sum(x) / len(x)
	
	raise NotImplementedError()