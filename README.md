# Fast-VSTM (Multi-level model acceleration of drug-target interaction prediction)

![Fast-VSTM Schematic](../ConPLex/assets/images/Fast-VSTM.png)

## Abstract
    The fast and accurate computational prediction of drug-target interactions (DTIs) is crucial for accelerating 
    the drug discovery process. While significant research has been conducted in this field, current methods 
    predominantly rely on large models to represent the features of drugs and targets, followed by predictions 
    using advanced machine learning techniques such as contrastive learning. However, as the drug molecule space 
    and target space continue to expand rapidly, predicting DTIs for real-world drug discovery faces substantial 
    computational challenges, making the acceleration of DTI prediction models a critical issue. To date, no solutions 
    have been proposed to address this challenge. In this paper, we propose a novel multi-level model acceleration 
    approach, Fast-VSTM. It first optimizes the ConPLex model, then applies a low-rank adaptive large model 
    fine-tuning approach for efficient optimization, and finally combines multi-head self-attention relation 
    distillation and feature distillation to lightweight the model and accelerate the DTI prediction 
    process.Experimental results show that Fast-VSTM achieves an average of 1.66x inference speedup on 
    multiple standard DTI datasets while maintaining comparable accuracy.


## Installation

### Install from PyPI

You should first have a version of [`cudatoolkit`](https://anaconda.org/nvidia/cudatoolkit) compatible with your system installed. Then run


### Compile from Source

```bash
git clone https://github.com/jiachengSigma/Fast-VSTM.git
cd Fast-VSTM
conda create -n VSTM python=3.9
conda activate VSTM
pip install -r requirements.txt
make poetry-download
export PATH="[poetry  install  location]:$PATH"
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
make install
```

## Usage

### Datasets and pre-trained models
Some datasets and pre-trained models are provided in the dataset and best_models folders.You can also download the dataset using the following command:

```bash
conplex-dti download --to datasets --benchmarks davis bindingdb biosnap biosnap_prot biosnap_mol dude
```


### Run benchmark training

| Parameter | Description | Default Value |
| --- | --- | --- |
| `--run-id` | Unique identifier for the experiment run. | No default value |
| `--wandb-proj` | Name of the Weights & Biases (wandb) project for logging experiment results. | No default value |
| `--wandb_save` | Whether to log experiment results to Weights & Biases. Requires login to wandb when enabled. | `False` |
| `--log-file` | Path to the log file for recording training information. | `./logs/scratch_testing.log` |
| `--model-save-dir` | Directory to save the best model during training. | `./best_models` |
| `--data-cache-dir` | Cache directory for downloading datasets. | `./datasets` |
| `--replicate` | Random seed for ensuring reproducibility of experiments. | `0` |
| `--device` | CUDA device number to use. For example, `0` refers to the first GPU. | `0` |
| `--verbosity` | Verbosity level of logging. Higher values mean more detailed logs. | `3` |
| `--task` | Name of the task to execute. Supported tasks include: `davis`, `bindingdb`, `biosnap`, `covid-19`. | No default value |
| `--contrastive-split` | Split method for contrastive learning, either `within` or `between`. | No default value |
| `--drug-featurizer` | Feature extractor for small molecule drug SMILES. Common options include: `UniMolFeaturizer`. | No default value |
| `--target-featurizer` | Feature extractor for protein sequences. Common options include: `ProtBertFeaturizer`. | No default value |
| `--model-architecture` | Name of the model architecture. Common options include: `SimpleCoembeddingNoSigmoid`. | No default value |
| `--latent-dimension` | Dimension size of the co-embedding space. | `1024` |
| `--latent-distance` | Distance metric used in the embedding space. Common options include: `Cosine`. | `Cosine` |
| `--epochs` | Total number of training epochs. | `50` |
| `--batch-size` | Batch size for binary classification tasks. | `32` |
| `--contrastive-batch-size` | Batch size for contrastive learning tasks. | `256` |
| `--shuffle` | Whether to shuffle the training data at the start of each epoch. | `True` |
| `--num-workers` | Number of worker threads for PyTorch DataLoader. | `0` |
| `--every-n-val` | Perform validation every N epochs. | `1` |
| `--lr` | Initial learning rate for binary classification tasks. | `1e-5` |
| `--lr-t0` | Reset learning rate to the initial value every T0 epochs in the learning rate annealing strategy. | `10` |
| `--contrastive` | Whether to enable contrastive learning. | `True` |
| `--clr` | Initial learning rate for contrastive learning tasks. | `1e-5` |
| `--clr-t0` | Reset learning rate to the initial value every T0 epochs for contrastive learning tasks. | `10` |
| `--margin-fn` | Margin function used in contrastive learning. Common options include: `tanh_decay`. | `tanh_decay` |
| `--margin-max` | Maximum margin value used in contrastive learning. | `0.25` |
| `--margin-t0` | Reset margin to the initial value every T0 epochs in the margin annealing strategy. | `10` |


```bash
python train.py \
    --run-id my_experiment \
    --wandb-proj NoSigmoidTest \
    --wandb_save \
    --log-file ./logs/scratch_testing.log \
    --model-save-dir ./best_models \
    --data-cache-dir ./datasets \
    --device 0 \
    --replicate 0 \
    --verbosity 3 \
    --task davis \
    --contrastive-split within \
    --drug-featurizer UnimolFeaturizer \
    --target-featurizer ProtBertFeaturizer \
    --model-architecture SimpleCoembeddingNoSigmoid \
    --latent-dimension 1024 \
    --latent-distance Cosine \
    --epochs 50 \
    --batch-size 32 \
    --contrastive-batch-size 256 \
    --shuffle \
    --num-workers 0 \
    --every-n-val 1 \
    --lr 1e-5 \
    --lr-t0 10 \
    --contrastive \
    --clr 1e-5 \
    --clr-t0 10 \
    --margin-fn tanh_decay \
    --margin-max 0.25 \
    --margin-t0 10
```
You can also conduct training through the following translation. Before the training, you need to edit the YAML file.


```bash
python -m VSTM.cli.train 
--config /config/default_config.yaml
```
### Accleartion by Fine-tuning Strategy 
We have fine-tuned the model using a variety of methods, specifically including Full FT, BitFit, HAdapter, PAdapter, LORA, and Adalora. You can modify the YAML file to carry out fine-tuning with different methods. Here, the default fine-tuning method in use is Adalora. It should be noted that when performing Full FT, the GPU memory needs to be large enough.

```bash
python -m VSTM.cli.fine_tune 
--config /config/finetune_config.yaml
```

### Use the VSTM with fine-tuning acceleration to conduct DTI (Drug-Target Interaction) prediction.

```bash
python -m VSTM.cli.predict 
--config /config/Adalora_config.yaml
```

Next, you can use the fine-tuned model to conduct Drug-Target Prediction.

```bash
python -m VSTM.cli.predict --data-file [pair predict file].tsv --model-path ./models/adalora_best_model.pt --outfile ./results.tsv
```

Format of `[pair predict file].tsv` should be `[protein ID]\t[molecule ID]\t[protein Sequence]\t[molecule SMILES]`

### Knowledge Distillation

```bash
cd VSTM/cli

python distill.py \
    --config configs/distill_config.yaml \
    --teacher_model prot_bert \
    --student_model student_tiny \
    --gpu 0
```

### Use Fast-VSTM to conduct DTI (Drug-Target Interaction) prediction.
```bash
python -m VSTM.cli.predict 
--config /config/Fast-VSTM.yaml

```



