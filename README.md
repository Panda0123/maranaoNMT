# Maranao NMT
Neural Machine Translation from Maranao Language to English and vice versa.
The model uses  [Text-To-Text Transfer Transformer (T5)](https://arxiv.org/abs/1910.10683) architecture.
Specifically, HuggingFace's [T5](https://huggingface.co/transformers/model_doc/t5.html) pytorch implementation was used.
The tokenizer used is the pre-trained [small-t5](https://huggingface.co/t5-small) for both languages.

## Table of Contents

* [Model](#model)
* [Test](#test-model)
* [Train](#train-model)
* [Finetune](#finetune-model)
* [Download](#download)
* [Dataset](#dataset)

### Model

There are two models: 
1. `t5-small` from HuggingFace which is finetuned on this task for 65 epochs. ~60M parameters. 
2. `t5-extra-small` which is trained using the hyperparameters [below](#train-model). ~39M parameters.

These models are trained on scraped data online.
The [dataset](clean_data/source) is pretty small and not of great quality.
The parameters and hyperparameters for `t5-extra-small` and training are in [model_train_config.py](src/model_train_config.py).

### Test Model
The easiest way to test the models is in [google colab](https://colab.research.google.com/drive/1zC4J25X7smDdEEse7Tt2gxzIE-vbNVWG?usp=sharing).
You may translate phrases/sentences by modifying corresponding variables as instructed in the notebook.

### Train Model
To train a model move to `src` directory and run `train.py` script while passing hyperparameters like described below or just edit `model_train_config.py`.
```
python train.py \
    --model=t5 \
    --num_layers=3 \
    --dropout_rate=0.1
    --num_epochs=65 \
    --batch_size=16 \
    --lr=1e-4 \
    --warmup_steps=500 \
    --num_workers=4 \
    --save_interval=659 \
    --log_interval=659 \
    --train_path="../clean_data/source/train_dataset/" \
    --validation_path="../clean_data/source/validation_dataset/"  \
    --model_path="../data/model/t5/" \
    --model_final_path="../data/model/final/t5/" \
    --logging_path="../data/model/final/t5/" \
    --load_model=False
```
If one of the arguments is not passed the default in [model_train_config.py](src/model_train_config.py) will be used.
As such you may just run `python train.py --model=t5` and use the default parameters.
Additionally, the structure of the dataset to be used for training must follow [this](#dataset).

### Download Pre-trained Model <div id='download'> </div>
The pre-trained model is currently hosted in [google drive](https://drive.google.com/file/d/1G2IJpmhUV9m0wJZbkHcta5Srl6z7x0VE/view?usp=sharing).
1. Download the zip file.
1. Unzip the file. The unzipped directory has the structure below.
> ``````
>MRN_NMT
>├── MRN_NMT_t5_extra_small
>│   ├── config.json
>│   ├── pytorch_model.bin
>│   └── training_args.bin
>└── MRN_NMT_t5_small
>     ├── config.json
>     ├── pytorch_model.bin
>     └── training_args.bin
>``````
3. Instantiate `model_final_path` in [model_train_config.py](src/model_train_config.py) to the relative path of the chosen 
model whether `MRN_NMT_t5_small` or `MRN_NMT_t5_extra_small` directory.

### Finetune Model
To finetune the pre-trained model, [download](#download) the model.
After instantiating the `model_final_path`, run `python train.py --model=t5 --loade_model=True`.
The --load_model=True arguement will load the model from `model_final_path` and train it.

### Dataset
The dataset is stored in `clean_data/source/`. The structure of training and validation dataset is demonstrated below.
The `tokenizer_dataset.txt` is optional only used for training a tokenizer.
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
       └── tokenizer_dataset.txt
``````
* The data must be a text file `.txt`. 
* The Maranao and English are separated.
* Each entry/line corresponds to 1 phrase/sentence.
* Line numbers specify their relation.
## TODO
* pretrain extra-small on scapred data using I.i.d. noise replace token.
    - unlike in the original they pretrained on mixture of unsupervised and superviesd  multi task.
* convert model to tensorflow, to tflite
* train a translation specific architecture.
