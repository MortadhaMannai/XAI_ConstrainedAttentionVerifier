# Regularization, semi-supervision, and supervision for a plausible attention-based explanation

Duc Hau Nguyen, Cyrielle Mallart, Guillaume Gravier, Pascale Sebillot

Code for the NLDB 2023 paper. Work partially funded by grant ANR-19-CE38-0011-03 from the French national research agency (ANR).

## Abstract

> Attention mechanism is contributing to the majority of recent advances in machine learning for natural language 
> processing. Additionally, it results in an attention map that shows the proportional influence of each input in its 
> decision. Empirical studies postulate that attention maps can be provided as an explanation for model output. However, 
> it is still questionable to ask whether this explanation helps regular people to understand and accept the model output 
> (the plausibility of the explanation). Recent studies show that attention weights in the RNN encoders are hardly 
> plausible because they spread on input tokens. We thus propose 3 additional constraints to the learning objective 
> function to improve the plausibility of the attention map: regularization to increase the attention weight sparsity, 
> semi-supervision to supervise the map by a heuristic and supervision by human annotation. Results show that all 
> techniques can improve the attention map plausibility at some level. We also observe that specific instructions for human 
> annotation might have a negative effect on classification performance. Beyond the attention map, the result of 
> experiments on text classification tasks also shows that no matter how the constraint brings the gain, the 
> contextualization layer plays a crucial role in finding the right space for finding plausible tokens.

## Project structure

* `src/` : Contains project source to reproduce results

  * `lstm_attention.py` : Main script for train and test
  * `summarize_result.py` : Summary all results to get average and confidential interval
  * `data/` : Pytorch datasets
  * `model/` :  Pytorch models
  * `data_module/` : Lightning modules for data, store logics of pre-post processing datasets
  * `model_module/` : Lightning modules for model, how to train model
  * `modules/` : all utility scripts

* `notebooks/` : Notebooks that generate figures and tables in the paper

* `figures/` : Contains generated figures in paper

* `requirements.cpu.txt` : replicate environment in CPU

* `requirements.gpu.txt` : replicate environment in GPU

## Software implementation

Experimentations are produced by `src/lstm_attention.py`, then average performances are synthetized by
`src/summarize_result.py`. Source code used to generate the results and figures in the paper are in the `notebooks`
folder.

## Dependencies

You'll need a working Python environment to run the code.

The recommended way to set up your environment is through the [Virtualenv Python](https://pypi.org/project/virtualenv/)
which provides the `virtual env`. The venv module supports creating lightweight “virtual environments”, each with their
own independent set of Python packages A virtual environment is created on top of an existing Python installation, known
as the virtual environment’s “base” Python, and may optionally be isolated from the packages in the base environment, so
only those explicitly installed in the virtual environment are available. (See further documentation
in [venv docs](https://docs.python.org/fr/3/library/venv.html)

The required dependencies are specified in the file `requirements.cpu.txt` (dependencies for CPU)
and `requirements.gpu.txt` (dependencies for GPU).

## Reproducing the results

1. Preparing dependencies for cpu from `requirements.gpu.txt`:

```bash
python -m venv eps
source eps/bin/activate
pip install -r requirements.cpu.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu113
```

2. Run an experimentation.
```bash
python src/single_lstm_attention_module.py \
          --cache .cache \
          --epoch 30 \
          --batch_size 128 \
          --vectors glove.840B.300d \
          --name hatexplain_supervise \
          --version run=2_lstm=1_lsup=0.6 \
          --data hatexplain \
          --lambda_supervise 0.6 \
          --n_lstm 1    
```

There 3 dataset for `--data`: `yelphat50`, `hatexplain`, `esnli`. 
Each technique has a corresponding lambda arguments:
  * **supervision**: `--lambda_supervise 0.5`
  * **regularization**: `--lambda_entropy 0.5`
  * **semisupervision**: `--lambda_heuristic 0.5` 

3. Summarize results for figures. It will automatically create a new folder `summary` in `--out_dir` path.

```bash
python src/summarize_result.py \
          --log_dir .cache \ 
          --out_dir .cache \ 
          --figure 
          --experiment hatexplain_supervise
```

## Citations

```latex
@InProceedings{10.1007/978-3-031-35320-8_20,
author="Nguyen, Duc Hau
and Mallart, Cyrielle
and Gravier, Guillaume
and S{\'e}billot, Pascale",
editor="M{\'e}tais, Elisabeth
and Meziane, Farid
and Sugumaran, Vijayan
and Manning, Warren
and Reiff-Marganiec, Stephan",
title="Regularization, Semi-supervision, and Supervision for a Plausible Attention-Based Explanation",
booktitle="Natural Language Processing and Information Systems",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="285--298",
abstract="Attention mechanism is contributing to the majority of recent advances in machine learning for natural language processing. Additionally, it results in an attention map that shows the proportional influence of each input in its decision. Empirical studies postulate that attention maps can be provided as an explanation for model output. However, it is still questionable to ask whether this explanation helps regular people to understand and accept the model output (the plausibility of the explanation). Recent studies show that attention weights in RNN encoders are hardly plausible because they spread on input tokens. We thus propose three additional constraints to the learning objective function to improve the plausibility of the attention map: regularization to increase the attention weight sparsity, semi-supervision to supervise the map by a heuristic and supervision by human annotation. Results show that all techniques can improve the attention map plausibility at some level. We also observe that specific instructions for human annotation might have a negative effect on classification performance. Beyond the attention map, results on text classification tasks also show that the contextualization layer plays a crucial role in finding the right space for finding plausible tokens, no matter how constraints bring the gain.",
isbn="978-3-031-35320-8"
}
```# XAI_ConstrainedAttentionVerifier
# XAI_ConstrainedAttentionVerifier
