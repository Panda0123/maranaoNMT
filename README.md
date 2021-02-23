# Maranao NMT
Neural Machine Translation from Maranao Language to English and vice versa.
The model uses  [Text-To-Text Transfer Transformer (T5)](https://arxiv.org/abs/1910.10683) architecture.
Specifically, HuggingFace's [T5](https://huggingface.co/transformers/model_doc/t5.html) pytorch implementation was used.
The pre-trained model (`t5-extra-small`) is trained in unsupervised way similar to the original paper (BERT-style: replacing spans and with noise density of 15% and span length of 3).
The tokenizer used is sentence piece and trained using [this](data/source/tokenizer_dataset.txt) data for both languages.

## Table of Contents
* [Model](#model)
* [Test](#test-model)
* [Train](#train-model)
* [Finetune](#finetune-model)
* [Download](#download)
* [Dataset](#dataset)

### Model

Pre-trained Model:
1. `t5-extra-small` - trained using the hyperparameters [below](#train-model). ~27M parameters

Two finetuned-model: 
1. `t5-small` from HuggingFace which is finetuned on this task. ~60M parameters. 
2. `t5-extra-small` the pre-trained model mentioned above which is also finetuned. ~27M parameters.

> Both are fine-tuned for:
> * 65 epochs 
> * batch size of 16
> * learning rate of 1e-4
> * warmup steps of 500
> * total of 42835 iterations

These models are trained on scraped data online. The [dataset](clean_data/source) is pretty small and not of great quality.

### Test Model
The easiest way to test the models is in [google colab](https://colab.research.google.com/drive/1zC4J25X7smDdEEse7Tt2gxzIE-vbNVWG?usp=sharing).
You may translate phrases/sentences by modifying corresponding variables as instructed in the notebook.

### Train Model
To train a model move to `src` directory and run `train.py` script while passing hyperparameters like described below or just edit `model_train_config.py`.
#### Pretraining
```
python train.py \
    --model=t5_pretraining \
    --num_layers=3 \
    --dropout_rate=0.1
    --num_epochs=70 \
    --batch_size=32 \
    --lr=1e-4 \
    --warmup_steps=500 \
    --num_workers=4 \
    --save_interval=3768 \
    --log_interval=3768 \
    --train_path="../clean_data/source/train_dataset/" \
    --validation_path="../clean_data/source/validation_dataset/"  \
    --model_path="../data/model/t5/" \
    --model_final_path="../data/model/final/t5/" \
    --logging_path="../data/model/final/t5/" \
    --load_model_path=None \
    --load_tokenizer_path="../clean_data/tokenizer/sp.model"
```
If one of the arguments is not passed the default in [model_train_config.py](src/model_train_config.py) will be used.
As such you may just run `python train.py --model=t5_pretraining` and use the default parameters.

### Finetune Model
To finetune the pre-trained model, [download](#download) the pre-trained model.
After instantiating the `load_model_path`, run `python train.py --model=t5_translation`.
Additionally, the structure of the dataset to be used for training must follow [this](#dataset).

### Download <div id='download'> </div>

#### Download Pre-trained Model 
The pre-trained model is currently hosted in [google drive](https://drive.google.com/file/d/1tvPS6OkRkGaLyCmfZpftH_G050WI133M/view?usp=sharing).
1. Download the zip file.
1. Unzip the file. The unzipped directory has the structure below.
> ``````
>MRN_Pretraining
> ├── config.json
> ├── pytorch_model.bin
> └── training_args.bin
>``````
3. Initialize `load_model_path` in [model_train_config.py](src/model_train_config.py) to the relative path of MRN_Pretraining.

#### Download Neural Machine Translation Model 
The NMT  model is currently hosted in [google drive](https://drive.google.com/file/d/1ZQcOaMBqrAbUMwvwqawJ53ndKInXlVlX/view?usp=sharing).
1. Download the zip file.
1. Unzip the file. The unzipped directory has the structure below.
> ``````
>MRN_NMT
>├── MRN_NMT_T5_small
>│   ├── config.json
>│   ├── pytorch_model.bin
>│   └── training_args.bin
>└── MRN_NMT_T5_extra_small
>    ├── config.json
>    ├── pytorch_model.bin
>    └── training_args.bin
>``````
3. Initialize `load_model_path` in [model_train_config.py](src/model_train_config.py) to the relative path of the chosen model whether `MRN_NMT_T5_small` or `MRN_NMT_T5_extra_small` directory.

### Dataset
The dataset is stored in `clean_data/source/`. The structure of training and validation dataset is demonstrated below. The `tokenizer_dataset.txt` is optional only used for training a tokenizer. The `mlm_training_dataset.txt` is used to pre-train the model.
``````
clean_data
   └── source
       ├── train_dataset
       │   ├── english.txt
       │   └── maranao.txt
       ├── validation_dataset
       │   ├── english.txt
       │   ├── english.txt
       │   └── maranao.txt
       ├── mlm_training_dataset.txt
       └── tokenizer_dataset.txt
``````
* The data must be a text file `.txt`. 
* The Maranao and English are separated.
* Each entry/line corresponds to 1 phrase/sentence.
* Line numbers specify their relation.

## TODO
* convert model to tensorflow, to tflite
* train a translation specific architecture.
