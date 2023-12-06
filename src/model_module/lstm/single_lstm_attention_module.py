import os
import warnings

from typing import Union

from os import path

import pytorch_lightning as pl
import torch

from torch import optim, nn
from torchtext.vocab.vectors import pretrained_aliases as pretrained

import torchmetrics as m

from model.lstm import SingleLstmAttention
from modules.const import Mode, SpecToken
from modules.logger import log
from modules import metrics, rescale, INF
from modules.loss import IoU, KLDivLoss


class SingleLSTMAttentionModule(pl.LightningModule):
	
	def __init__(self, cache_path, mode, vocab, pretrained_vectors: Union[str, torch.tensor]=None,
	             lambda_entropy:float = 0.,
	             lambda_supervise:float = 0.,
	             lambda_lagrange:float = 0.,
	             lambda_heuristic:float = 0.,
	             data='Unk data',
	             num_class=-1,
	             n_context=1,
	             concat_context=False,
	             **kwargs):
		super(SingleLSTMAttentionModule, self).__init__()
		
		# log hyperparameters into hparams.yaml
		self.save_hyperparameters('data', 'n_context', 'concat_context', 'lambda_entropy', 'lambda_supervise', 'lambda_lagrange', 'lambda_heuristic')
		self.data = data
		
		if pretrained_vectors is not None and isinstance(pretrained_vectors, str):
			vector_path = path.join(cache_path, '..', '.vector_cache')
			os.makedirs(vector_path, exist_ok=True)
			vectors = pretrained[pretrained_vectors](cache=vector_path)
			pretrained_vectors = [vectors[token] for token in vocab.get_itos()]
			pretrained_vectors = torch.stack(pretrained_vectors)

		self.model = SingleLstmAttention(pretrained_embedding=pretrained_vectors,
		                                 vocab_size=len(vocab),
		                                 d_embedding=kwargs['d_embedding'],
		                                 padding_idx=vocab[SpecToken.PAD],
		                                 n_class=num_class,
		                                 n_lstm=n_context,
		                                 concat_context=concat_context,
		                                 attention_raw=True)
		
		self.loss_fn = nn.CrossEntropyLoss()
		self.supervise_loss_fn = IoU()
		self.lagrange_loss_fn = nn.MSELoss()
		self.heuristic_loss_fn = KLDivLoss(reduction='batchmean', log_target=True)
		
		self.num_class = num_class
		self.vocab = vocab
		self._mode = mode
		self.lambda_entropy = lambda_entropy
		self.lambda_supervise = lambda_supervise
		self.lambda_lagrange = lambda_lagrange
		self.lambda_heuristic = lambda_heuristic
		
		self.caching_weight = None
		
		template_y_metrics = m.MetricCollection({
			'y:accuracy': m.Accuracy(num_classes=num_class, multiclass=True),
			'y:fscore': m.F1Score(num_classes=num_class, multiclass=True)
		})
		
		with warnings.catch_warnings():
			template_attention_metrics = m.MetricCollection({
				'a:AUROC': m.AUROC(average='micro'),
				'a:AUPRC': m.AveragePrecision(average='micro'),
				'a:AURecall': metrics.AURecall(),
				'a:AUPrecision': metrics.AUPrecision(),
				'a:Jaccard': metrics.PowerJaccard(),
				'a:Specificity': m.Specificity(),
				'a:Dice': m.Dice(),
				'a:IoU': m.JaccardIndex(num_classes=2),
			})
			warnings.simplefilter("ignore")
		
		PHASES = ['TRAIN', 'VAL', 'TEST']
		self.y_metrics = nn.ModuleDict({
			phase: template_y_metrics.clone() for phase in PHASES
		})
		self.attention_metrics = nn.ModuleDict({
			phase: template_attention_metrics.clone() for phase in PHASES
		})
		self.entropy_metric = nn.ModuleDict({
			phase: metrics.Entropy(normalize=True) for phase in PHASES
		})
		self.reg_term_metric = nn.ModuleDict({
			phase: m.MeanMetric() for phase in PHASES
		})
		
	def forward(self, ids, mask):
		return self.model(ids=ids, mask=mask)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters())
		return optimizer
	
	def training_step(self, batch, batch_idx, val=False):
		
		y_true = batch['y_true']
		padding_mask = batch['padding_mask']
		a_true = batch['a_true']
		# a_true_entropy = batch['a_true_entropy']
		heuristic = batch['heuristic']
		y_hat, a_hat = self(
			ids=batch['token_ids'],
			mask=padding_mask
		)
		
		loss_classif = self.loss_fn(y_hat, y_true)
		a_hat_entropy = metrics.entropy(a_hat, padding_mask, normalize=True)
		loss_entropy = a_hat_entropy.mean()
		
		# Sigmoid for IoU loss
		flat_a_hat = self.flatten_attention(attention=a_hat, condition=y_true > 0, pad_mask=padding_mask, normalize='sigmoid')
		flat_a_true = self.flatten_attention(attention=a_true, condition=y_true > 0, pad_mask=padding_mask)

		if flat_a_true is None:
			loss_supervise = torch.tensor(0.).type_as(loss_classif)
		else:
			loss_supervise = self.supervise_loss_fn(flat_a_hat, flat_a_true)

		loss_heuristic = self.heuristic_loss_fn(a_hat.log_softmax(dim=1), heuristic)
		
		if loss_heuristic > 10:
			log.debug(f'loss_heuristic={loss_heuristic}')

		#
		
		# flat_a_hat_sig = self.flatten_attention(attention=a_hat, condition=y_true > 0, pad_mask=padding_mask, normalize='sigmoid')
		# flat_a_true = self.flatten_attention(attention=a_true, condition=y_true > 0, pad_mask=padding_mask)
		#
		# flat_a_hat_soft = self.flatten_attention(attention=a_hat, condition=y_true > 0, pad_mask=padding_mask, normalize='softmax')
		# flat_a_heuristic = self.flatten_attention(attention=heuristic, condition=y_true > 0, pad_mask=padding_mask)
		#
		# if flat_a_true is None:
		# 	loss_supervise = torch.tensor(0.).type_as(loss_classif)
		# 	loss_heuristic = torch.tensor(0.).type_as(loss_classif)
		# else:
		# 	loss_supervise = self.supervise_loss_fn(flat_a_hat_sig, flat_a_true)
		# 	loss_heuristic = self.heuristic_loss_fn(flat_a_hat_soft, flat_a_heuristic)
		
		# loss_lagrange = self.lagrange_loss_fn(a_hat_entropy, a_true_entropy)
		
		# loss = loss_classif + self.lambda_entropy * loss_entropy \
		#        + self.lambda_supervise * loss_supervise \
		#        + self.lambda_lagrange * loss_lagrange \
		#        + self.lambda_heuristic * loss_heuristic
		
		loss = loss_classif + self.lambda_entropy * loss_entropy \
		       + self.lambda_supervise * loss_supervise \
		       + self.lambda_heuristic * loss_heuristic
		
		return {'loss': loss,
		        'loss_entropy': loss_entropy,
		        'loss_supervise': loss_supervise,
		        # 'loss_lagrange': loss_lagrange,
		        'loss_heuristic': loss_heuristic,
		        'y_hat': y_hat, 'y_true': y_true, 'a_hat': a_hat, 'a_true': a_true, 'padding_mask': padding_mask}
	
	def validation_step(self, batch, batch_idx):
		return self.training_step(batch, batch_idx, True)
	
	def test_step(self, batch, batch_idx, dataloader_idx=None):

		y_hat, a_hat = self(
			ids=batch['token_ids'],
			mask=batch['padding_mask']
		)
		
		a_hat = a_hat.detach()
		
		return {'y_hat': y_hat,
		        'y_true': batch['y_true'],
		        'a_hat': a_hat,
		        'a_true': batch['a_true'],
		        'padding_mask': batch['padding_mask']}
	
	def predict_step(self, batch, batch_idx):
		
		padding_mask = batch['padding_mask']
		y_hat, a_hat = self(
			ids=batch['token_ids'],
			mask=batch['padding_mask']
		)
		a_hat = a_hat.detach()
		a_hat = self.model.softmax(a_hat)
		
		return {'y_hat': y_hat.argmax(axis=-1),
		        'a_hat': a_hat,
		        'padding_mask': padding_mask}
	
	def step_end(self, outputs, stage: str = 'TEST'):
		
		a_hat, a_true = outputs['a_hat'], outputs['a_true']
		y_hat, y_true = outputs['y_hat'], outputs['y_true']
		padding_mask = outputs['padding_mask']
		
		flat_a_hat = self.flatten_attention(attention=a_hat, condition=y_true > 0, pad_mask=padding_mask, normalize='softmax_rescale')
		flat_a_true = self.flatten_attention(attention=a_true, condition=y_true > 0, pad_mask=padding_mask)
		
		# log attentions metrics
		if flat_a_hat.size(0) > 0:
			metric_a = self.attention_metrics[stage](flat_a_hat, flat_a_true)
			metric_a['a:entropy'] = self.entropy_metric[stage](a_hat, padding_mask)
			metric_a = {f'{stage}/{k}': v.item() for k, v in metric_a.items()}  # put metrics within same stage under the same folder
			self.log_dict(metric_a, prog_bar=True)
		
		# log for classification metrics
		metric_y = self.y_metrics[stage](y_hat, y_true)
		metric_y = {f'{stage}/{k}': v for k, v in metric_y.items()}  # put metrics within same stage under the same folder
		self.log_dict(metric_y, prog_bar=True)
		
		if stage != 'TEST':
			# if not in test stage, log loss metrics
			loss_names = [k for k in outputs.keys() if 'loss' in k]
			for loss_metric in loss_names:
				self.log(f'{stage}/{loss_metric}', outputs[loss_metric], prog_bar=True, sync_dist=True)
	
	def training_step_end(self, outputs):
		return self.step_end(outputs, stage='TRAIN')
	
	def validation_step_end(self, outputs):
		return self.step_end(outputs, stage='VAL')
	
	def test_step_end(self, outputs):
		return self.step_end(outputs, stage='TEST')
	
	def flatten_attention(self, attention, pad_mask, condition=None, normalize: str = ''):
		"""Filter attention
		
		Parameters
		----------
		attention :
		pad_mask :
		condition :
		normalize : softmax, softmax_rescale, sigmoid

		Returns
		-------

		"""
		
		if condition is None:
			condition = torch.ones(attention.size(0)).type(torch.bool)
		
		if (~condition).all():
			return torch.tensor([])
		
		# Filter by condition on y
		attention = attention[condition]
		pad_mask = pad_mask[condition]
		
		# Ir normalize specify:
		if normalize == 'sigmoid':
			attention = torch.sigmoid(attention)
		if 'softmax' in normalize:
			attention = torch.softmax(attention + pad_mask.float() * -INF, dim=1)
		if 'rescale' in normalize:
			attention = rescale(attention, pad_mask)
		
		# Filter by mask
		return attention[~pad_mask]
	
	def on_train_start(self):   
		init_hp_metrics = {f'TEST/{k}': 0 for k in self.y_metrics['TEST']}
		init_hp_metrics.update({f'TEST/{k}': 0 for k in self.attention_metrics['TEST']})
		init_hp_metrics.update({f'TEST/a:entropy': 0})
		self.logger.log_hyperparams(self.hparams, init_hp_metrics)
	
	def on_train_epoch_start(self):
		# Make new line for progress bar.
		# See: https://github.com/PyTorchLightning/pytorch-lightning/issues/2189
		if self._mode == Mode.DEV:
			print()
	
	def epoch_end(self, stage):
		if self._mode == Mode.EXP:
			metric = self.y_metrics[stage].compute()
			try:
				metric.update(self.attention_metrics[stage].compute())
			except RuntimeError as e:
				log.error(e)
				
			metric.update({
				'a:entropy': self.entropy_metric[stage].compute()
			})
			metric = {k: round(v.item(), 3) for k, v in metric.items()}
			log.info(f'Epoch {self.current_epoch} {stage}:{metric}')
			
	def on_train_epoch_end(self):
		return self.epoch_end('TRAIN')
		
	def on_validation_epoch_end(self):
		return self.epoch_end('VAL')
		
	def __str__(self):
		return str(self.model)

