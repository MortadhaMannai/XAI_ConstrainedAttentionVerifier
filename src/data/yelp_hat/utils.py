import re

from bs4 import BeautifulSoup

from modules.logger import log
import numpy as np


def yelp_hat_ham(html, spacy_model):
	soup = BeautifulSoup(html, 'html.parser')
	tags = [tag for tag in soup.find_all('span') if tag.string is not None]
	
	tag_annot = [int('active' in t.get('class', [])) for t in tags]
	tag_str = [str(t.string) for t in tags]
	
	ham = []
	
	for annot, splitted_tokens in zip(tag_annot, spacy_model.pipe(tag_str, disable=['ner',"parser"])):
		annotation = [annot * int(not tk.is_punct) for tk in splitted_tokens]
		ham += annotation
	
	return ham

def yelp_hat_token(html, spacy_model, lemma=True, lower=True):
	soup = BeautifulSoup(html, 'html.parser')
	tags = [str(tag.string) for tag in soup.find_all('span') if tag.string is not None]
	
	tokens = [tk for doc in spacy_model.pipe(tags, disable=['ner', "parser"]) for tk in doc]
	
	if lemma:
		tokens = [tk.lemma_ for tk in tokens]
		
	if lower:
		tokens = [tk.lower() for tk in tokens]
	
	if not lemma and not lower:
		tokens = [tk.text for tk in tokens]
	
	return tokens


def cam(row):
	return np.logical_and(row['ham_0'], row['ham_1'], row['ham_2'])


def sam(row):
	return np.logical_or(row['ham_0'], row['ham_1'], row['ham_2'])


def ham(row):
	return ((row['ham_0'] + row['ham_1'] + row['ham_2']) / 3 >= 0.5).astype(int)


def generate_binary_human_attention_vector(html, num_words_in_review, max_words):
	# Function provided by the dataset :
	# https://github.com/cansusen/Human-Attention-for-Text-Classification/blob/master/generate_ham/sample_generate.ipynb
	
	p = re.compile('<span(.*?)/span>')
	all_span_items = p.findall(html)
	
	if html == '{}':
		log.error('Empty human annotation - This should never print')
		return [0] * max_words
	
	if len(all_span_items) == num_words_in_review + 1:
		if (all_span_items[num_words_in_review] == '><') or (all_span_items[num_words_in_review] == ' data-vivaldi-spatnav-clickable="1"><'):
			
			binarized_human_attention = [0] * max_words
			for i in range(0, len(all_span_items) - 1):
				if 'class="active"' in all_span_items[i]:
					binarized_human_attention[i] = 1
		
		else:
			log.error('This should never print.')
	else:
		log.error('This should never print.')
	
	return binarized_human_attention