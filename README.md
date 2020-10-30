# Investigating Pretrained Language Models for Graph-to-Text Generation

This repository contains the code for the paper: "[Investigating Pretrained Language Models for Graph-to-Text Generation](https://arxiv.org/pdf/2007.08426.pdf)".

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

This project is implemented using the framework [HuggingFace](https://huggingface.co/). Please, refer to their websites for further details on the installation and dependencies.

## Environments and Dependencies

- python 3.6
- transformers 3.3.1
- pytorch-lightning 0.9.0
- torch 1.4.0
- parsimonious 0.8.1
## Datasets

In our experiments, we use the following datasets: [AMR17](https://catalog.ldc.upenn.edu/LDC2017T10), [WebNLG](https://webnlg-challenge.loria.fr/challenge_2017/) and [AGENDA](https://github.com/rikdz/GraphWriter/tree/master/data).

## Preprocess

First, convert the dataset into the format required for the model.

For the AMR17, run:
```
./preprocess_AMR.sh <dataset_folder>
```

For the WebNLG, run:
```
./preprocess_WEBNLG.sh <dataset_folder>
```

For the AGENDA, run:
```
./preprocess_AGENDA.sh <dataset_folder>
```


## Finetuning

For finetuning the models using the AMR dataset, execute:
```
./finetune_AMR.sh <model> <gpu_id>
```

For the WebNLG dataset, execute:
```
./finetune_WEBNLG.sh <model> <gpu_id>
```

For the AGENDA dataset, execute:
```
./finetune_AGENDA.sh <model> <gpu_id>
```
 
Options for `<model>` are `t5-small`, `t5-base`, `t5-large`, `facebook/bart-base` or `facebook/bart-large`. 

Example:
```
./finetune_AGENDA.sh t5-small 0
```


## Decoding

For decoding, run:
```
./decode_AMR.sh <model> <checkpoint> <gpu_id>
./decode_WEBNLG.sh <model> <checkpoint> <gpu_id>
./decode_AGENDA.sh <model> <checkpoint> <gpu_id>
```

Example:
```
./decode_WEBNLG.sh t5-base webnlg-t5-base.ckpt 0
```

## Trained models

| AMR17          |
| :------------- |
| [bart-base](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/amr-bart-base.ckpt) - BLEU: 36.71 ([output]()) |
| [bart-large](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/amr-bart-large.ckpt) - BLEU: 43.47 ([output]()) |
|  [t5-small](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/amr-t5-small.ckpt) - BLEU: 38.45 ([output]()) | 
| [t5-base](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/amr-t5-base.ckpt) - BLEU: 42.54 ([output]())  |
| [t5-large](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/amr-t5-large.ckpt) - BLEU: 45.80 ([output]()) |

| WebNLG   | 
| :------------- |
| [bart-base](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/webnlg-bart-base.ckpt) - All BLEU: 53.11 ([output]()), Seen BLEU: 62.74 ([output]()), Unseen BLEU: 41.53 ([output]()) | 
| [bart-large](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/webnlg-bart-large.ckpt) - All BLEU: 54.72 ([output]()), Seen BLEU: 63.45 ([output]()), Unseen BLEU: 43.97 ([output]()) |
| [t5-small](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/webnlg-t5-small.ckpt) - All BLEU: 56.34 ([output]()), Seen BLEU: 65.05 ([output]()), Unseen BLEU: 45.37 ([output]()) | 
| [t5-base](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/webnlg-t5-base.ckpt) - All BLEU: 59.17 ([output]()), Seen BLEU: 64.64 ([output]()), Unseen BLEU: 52.55 ([output]()) | 
| [t5-large](https://public.ukp.informatik.tu-darmstadt.de/ribeiro/graph2text/webnlg-t5-large.ckpt) - All BLEU: 59.70 ([output]()), Seen BLEU: 64.71 ([output]()), Unseen BLEU: 53.67 ([output]()) | 

| AGENDA   | 
| :------------- |
| coming soon |

\* BLEU values for AMR17 are calculated using [sacreBLEU](https://github.com/mjpost/sacrebleu) in detok outputs. BLEU values for WebNLG are calculated using tok outputs using the [challange's script](https://gitlab.com/webnlg/webnlg-baseline), that uses multi-bleu.perl.



## More
For more details regading hyperparameters, please refer to [HuggingFace](https://huggingface.co/).


Contact person: Leonardo Ribeiro, ribeiro@aiphes.tu-darmstadt.de

## Citation
```
@misc{ribeiro2020investigating,
      title={Investigating Pretrained Language Models for Graph-to-Text Generation}, 
      author={Leonardo F. R. Ribeiro and Martin Schmitt and Hinrich Schütze and Iryna Gurevych},
      year={2020},
      eprint={2007.08426},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```