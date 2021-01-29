# Maranao NMT
Neural Machine Translation from Maranao Language to English and vice versa.
The model uses  [Text-To-Text Transfer Transformer (T5)](https://arxiv.org/abs/1910.10683) architecture.
Specifically, HuggingFace's [T5](https://huggingface.co/transformers/model_doc/t5.html) pytorch implementation was used.
The tokenizer used is the pre-trained [small-t5](https://huggingface.co/t5-small) for both language.

## Table of Contents

* [Model](#model)
* [Test](#test-model)
* [Train](#train-model)
* [Finetune](#finetune-model)
* [Download](#download)
    * [Respository](#download-repository)
    * [Pre-trained Model](#download-model)
* [Dataset](#dataset)

### Model

The pre-trained model is trained on scraped data online.
The [dataset](clean_data/source) is pretty small and not of great quality as such the model is trained with 3 layers only for both the encoder and decoder.
Model's  can be modified. The trained model has  ~`39M` parameters.
The parameters and hyperparameters are in [config.py](#src/config.py).

### Test Model
The easiest way to test the model is in [google colab](https://colab.research.google.com/drive/1zC4J25X7smDdEEse7Tt2gxzIE-vbNVWG?usp=sharing).
You may translate phrase/senentece by modifying corresponding variables as instructed in the notebook.

### Train Model
To train a model move to `src` directory and run `train.py` script while passing parameters like described below or just edit `train_config.py`.
```
python train.py \
    --model=t5 \
    --num_layers=3 \
    --dropout_rate=0.1
    --num_epochs=60 \
    --batch_size=16 \
    --lr=1e-4 \
    --warmup_steps=500 \
    --num_workers=4 \
    --save_interval=600 \
    --log_interval=600 \
    --train_path="../clean_data/source/train_dataset/" \
    --validation_path="../clean_data/source/validation_dataset/"  \
    --model_path="../data/model/t5/" \
    --model_final_path="../data/model/final/t5/" \
    --logging_path="../data/model/final/t5/" \
    --load_model=False
```
If one of the argument is not passed it will use the default in [config.py](src/config.py).
As such you may just run `python train.py` and use the default parameters in [config.py](src/config.py).
Additionally, the structure of the dataset to be used for training must follow [this](#dataset).

### Download Pre-trained Model <div id='download'> </div>
The pre-trained model is currently hosted in [google drive](https://drive.google.com/drive/folders/1be4kGVViFSPMh2ZnhJ_gxyWXrmMVolGd).
1. Download the three files and store them in one directory.
1. Instantiate `model_final_path` in [config.py](#src/config.py) to the relative path of that directory.

### Finetune Model
To finetune the pre-trained model, [download](#download) the model.
After instantiating the `model_final_path`, run `python train.py --loade_model=True`.
This will continue the training of the model.

### Dataset
The dataset is stored in `clean_data/source/`. The structure of training and validation dataset is demostrated below.
``````
├── clean_data
   ├── source
      ├── train_dataset
      │   ├── english.txt
      │   └── maranao.txt
      └── validation_dataset
          ├── english.txt
          └── maranao.txt
``````
* The data must be a text file `.txt`. 
* The Maranao and English are separated.
* Each entry/line corresponds to 1 phrase/sentece.
* Line number specify their relation.

## TODO
* script for finetune
* train a translation specific architecture.

maranaoNMT:
    - model
        - T5
    - trainer
