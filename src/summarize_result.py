import json
import os
from os import path
from argparse import ArgumentParser

import pandas as pd
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from yaml import SafeLoader

from modules.logger import log, init_logging

_FIGURES = 'figures'
_TABLE = 'table'

def parse_argument(prog: str = __name__, description: str = 'Summarize results produced by experiments'):
	"""Parse arguments passed to the script.
	
	Parameters
	----------
	prog : str
		name of the programme (experimentation)
	description : str
		What do we do to this script

	Returns
	-------
	args :
		argument values as dictionary
	"""
	parser = ArgumentParser(prog=prog, description=description)
	
	# Optional stuff
	
	# Summary arguments
	parser.add_argument('--summary', action='store_true', help='Force to regenerate summary cache')
	parser.add_argument('--experiment', type=str, nargs='+', help='Experiment name in log_dir. '
	                                                                'If not given, try to summary all experiments'
	                                                                'If more then 2 values, try to generate appropriate scaling '
	                                                                'between different experiments')
	
	parser.add_argument('--all', action='store_true', help='Generate all features from summary')
	
	# Figure arguments
	parser.add_argument('--figure', action='store_true', help='Regenerate figures')
	parser.add_argument('--ylim', type=float, nargs=2, help='Scale for y-axis in figure')
	
	# Table arguments
	parser.add_argument('--table', action='store_true', help='Force to regenerate summary table')
	parser.add_argument('--round', type=int, help='Rounding table in latex')
	
	# General arguments
	parser.add_argument('--log_dir', '-l', type=str, required=True, help='Path where logs is saved, contains different experimentations set up.')
	parser.add_argument('--out_dir', '-o', type=str, required=True, help='Output directory')
	return parser.parse_args()

if __name__ == '__main__':
	
	args = parse_argument()
	init_logging(color=True)
	
	log_dir = args.log_dir
	
	if isinstance(args.experiment, list) and len(args.experiment) > 0:
		
		for e in args.experiment:
			if args.summary and not path.exists(path.join(log_dir, e)): raise ValueError(f'Experiment {e} does not exist in {log_dir}')
		
	else:
		args.experiment = [e for e in os.listdir(log_dir) if '.DS_Store' not in e]
	
	# summary results
	summary = [None] * len(args.experiment)
	
	for experiment in args.experiment:

		experiment_path = path.join(log_dir, experiment)
		out_path = path.join(args.out_dir, 'summary', experiment)
		parquet_path = path.join(out_path, 'summary.parquet')
		results = list()
		runs = list(os.listdir(experiment_path))
		runs = [r for r in runs if '.DS_Store' not in r]
		runs = [r for r in runs if path.exists(path.join(experiment_path, r, 'hparams.yaml'))]
		runs = [r for r in runs if path.exists(path.join(experiment_path, r, 'score.json'))]
		
		if len(runs) == 0: continue
		
		log.info(f'Summarize {experiment_path}')
		
		if args.summary or not path.exists(parquet_path) or not path.exists(parquet_path):
		
			for run in tqdm(runs, total=len(runs), desc=f'Summarize {experiment}'):
				
				with open(path.join(experiment_path, run, 'hparams.yaml')) as f:
					hparam = yaml.load(f, Loader=SafeLoader)
				
				with open(path.join(experiment_path, run, 'score.json')) as f:
					score = json.load(f)
					score = {k.replace('TEST/', ''): v for k, v in score.items()}
				
				score_row = {**hparam, **score}
				results.append(score_row)
			
			df = pd.DataFrame(results, index=runs)
			
			# Save to summary
			## Cache dataframe
			os.makedirs(out_path, exist_ok=True)
			df.to_parquet(parquet_path)
		
		else:
			df = pd.read_parquet(parquet_path)
			with open(path.join(experiment_path, runs[0], 'hparams.yaml')) as f:
				hparam = yaml.load(f, Loader=SafeLoader)
			
			with open(path.join(experiment_path, runs[0], 'score.json')) as f:
				score = json.load(f)
				score = {k.replace('TEST/', ''): v for k, v in score.items()}
		
		figure_path = path.join(out_path, _FIGURES)
		table_path = path.join(out_path, _TABLE)
		
		print_figure = not path.exists(figure_path) or args.figure
		print_table = not path.exists(table_path) or args.table
		if print_figure: log.debug(f'Print figure')
		if print_table: log.debug(f'Print table')
		
		## Save figure
		objectives = [p for p in hparam.keys() if 'lambda' in p and df[p].nunique() > 1] # ["lambda_entropy", "lambda_supervise"]
		groups = [p for p in hparam.keys() if 'lambda' not in p and df[p].nunique() > 1] # ["n_lstm", "n_attention", "n_cnn"]
		sns.set(font_scale=2)
		
		os.makedirs(figure_path, exist_ok=True)
		os.makedirs(table_path, exist_ok=True)
		
		for metric in score.keys():
			for lambda_value in objectives:
				for g in groups:
					
					fig_path = path.join(figure_path, f'x={lambda_value}_y={metric}_color={g}.png')
					latex_path = path.join(table_path, f'{experiment}_{g}_{lambda_value}.tex')
					html_path = path.join(table_path, f'{experiment}_{g}_{lambda_value}.html')
					
					if print_figure:
						fig = plt.figure(figsize=(20, 15), clear=True)
						graphic = sns.pointplot(data=df, x=lambda_value, y=metric, hue=g, errwidth=2, capsize=0.1, dodge=True, palette='husl')
						if args.ylim: graphic.set_ylim(tuple(args.ylim))
						plt.savefig(fig_path, bbox_inches="tight")
						# log.debug(f'saved at {fig_path}')
						plt.close(fig)
			
					if print_table:
						mean = df.groupby([g, lambda_value]).mean(numeric_only=False)
						std = df.groupby([g, lambda_value]).std(numeric_only=False)
						
						if args.round:
							mean = mean.round(args.round)
							std = std.round(args.round)
						
						df_quantify = mean.astype(str) + u"\u00B1" + std.astype(str)
						with open(latex_path, 'w') as f:
							f.write(df_quantify.style.to_latex())
						
						with open(html_path, "w") as f:
							f.write(df_quantify.to_html(justify='center'))
			
