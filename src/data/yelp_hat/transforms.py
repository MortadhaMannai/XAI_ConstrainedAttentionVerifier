import spacy
from bs4 import BeautifulSoup
from torch.nn import Module


class BinarizingAnnotationTransform(Module):
	
	def __init__(self, spacy_model):
		"""Binarize annotations into binary content
		
		Parameters
		----------
		spacy_model :
		"""
		self.sm = spacy_model
		
	def forward(self, html):
		soup = BeautifulSoup(html, 'html.parser')
		tags = [tag for tag in soup.find_all('span') if tag.string is not None]
		
		tag_annot = [int('active' in t.get('class', [])) for t in tags]
		tag_str = [str(t.string) for t in tags]
		
		ham = []
		
		for annot, splitted_tokens in zip(tag_annot, self.sm.pipe(tag_str, disable=['ner', "parser"])):
			annotation = [annot * int(not tk.is_punct) for tk in splitted_tokens]
			ham += annotation
		
		return ham


class SpacyTokenizeTransform(Module):
	
	def __init__(self, spacy_model: spacy.Language):
		"""Tokenize with spacy tokenizer
		
		Parameters
		----------
		spacy_model : spacy.Language
		"""
		self.sm = spacy_model
	
	def forward(self, html):
		"""
		
		Parameters
		----------
		html : str
			HTML text from YelpHat
			
		Returns
		-------
		list of int
			list of binary, 1 if word is annotated, 0 otherwise.
		"""
		soup = BeautifulSoup(html, 'html.parser')
		tags = [tag for tag in soup.find_all('span') if tag.string is not None]
		
		tag_annot = [int('active' in t.get('class', [])) for t in tags]
		tag_str = [str(t.string) for t in tags]
		
		ham = []
		
		for annot, splitted_tokens in zip(tag_annot, self.sm.pipe(tag_str, disable=['ner', "parser"])):
			annotation = [annot * int(not tk.is_punct) for tk in splitted_tokens]
			ham += annotation
		
		return ham