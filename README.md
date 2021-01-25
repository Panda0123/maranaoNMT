# Maranao NMT
Neural Machine Translation from Maranao Language to English and vice versa.
The model uses the [Text-To-Text Transfer Transformer (T5)](https://arxiv.org/abs/1910.10683) architecture.
Specifically, HuggingFace's [small-t5](https://huggingface.co/t5-small) implementation was used.


## Model

The model is trained on scraped data online.
The [dataset](clean_data/source) is pretty small and not of great quality as such the model is trained with 3 layers only for both the encoder and decoder.
Due to limitation of data the dataset is split by 95% trainign and 5% evaluation.
The model can be downloaded from [google drive](https://drive.google.com/drive/folders/1be4kGVViFSPMh2ZnhJ_gxyWXrmMVolGd).
The parameters of the model and training are listed below.

```py
num_epochs = 60
learning_rate = 1e-4
warmup_steps = 500
num_layers = 3 # both encoder and decoder
dropout_rate = 0.1
# default parameters
```

## Test Model
You may test the model in [google colab](https://colab.research.google.com/drive/1zC4J25X7smDdEEse7Tt2gxzIE-vbNVWG#scrollTo=FlfbqTkieaAa).
Just run the cells from top to bottom. You may enter input by modifying corresponding variables as instructed in the notebook.


## TODO
1. finish documentation.
