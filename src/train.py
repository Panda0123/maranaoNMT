import argparse
import os

import model_train_config
import tokenizer_config
import tokenizer
from model.t5.train_translation import  run as trainT5Translation
from model.t5.train_mlm import run  as trainT5Pretraining

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def run(
    model: str,
    num_layers: int,
    dropout_rate: float,
    num_epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    num_workers: int,
    save_interval: int,
    log_interval: int,
    train_path: str,
    validation_path: str,
    model_path: str,
    model_final_path: str,
    logging_path: str,
    load_model_path: str,
    tokenizer_dataset_path: str,
    tokenizer_path: str,
    load_tokenizer_path: str,
    vocab_size: int,
):
    trainer = None
    if model == "sp":
        trainer = tokenizer.trainSP(tokenizer_dataset_path,
                                    tokenizer_path,
                                    vocab_size,
                                    modelPrefix=model)
    elif model == "t5_pretraining":  # masked language modelling and replacing noise tokens
        trainer = trainT5Pretraining(num_layers,
                                      dropout_rate,
                                      num_epochs,
                                      batch_size,
                                      lr,
                                      warmup_steps,
                                      num_workers,
                                      save_interval,
                                      log_interval,
                                      train_path,
                                      model_path,
                                      model_final_path,
                                      logging_path,
                                      load_model_path=load_model_path,
                                      load_tokenizer_path=load_tokenizer_path)
    elif model == "t5_translation":
        trainer = trainT5Translation(num_layers,
                                      dropout_rate,
                                      num_epochs,
                                      batch_size,
                                      lr,
                                      warmup_steps,
                                      num_workers,
                                      save_interval,
                                      log_interval,
                                      train_path,
                                      validation_path,
                                      model_path,
                                      model_final_path,
                                      logging_path,
                                      load_model_path,
                                      load_tokenizer_path)

    return trainer


def createArgumentParser():
    parser = argparse.ArgumentParser(prog="Trainer")
    # model
    parser.add_argument("--model",
                        help="name of the model default to t5",
                        type=str)
    parser.add_argument("--num_layers",
                        help="number of layers for both encoder and decoder",
                        type=int)
    parser.add_argument("--dropout_rate",
                        type=float)

    # training
    parser.add_argument("--num_epochs",
                        type=int)
    parser.add_argument("--batch_size",
                        type=int)
    parser.add_argument("--lr",
                        type=float)
    parser.add_argument("--warmup_steps",
                        type=int)
    parser.add_argument("--num_workers",
                        help="number of dataloader workers",
                        type=int)
    parser.add_argument("--save_interval",
                        help="number of steps between two save checkpoint",
                        type=int)
    parser.add_argument("--log_interval",
                        help="number of steps between two logs",
                        type=int)

    # paths
    parser.add_argument("--train_path",
                        help="relative path to the train dataset",
                        type=str)
    parser.add_argument("--validation_path",
                        help="relative path to the valid dataset",
                        type=str)
    parser.add_argument("--model_path",
                        help="relative path to where model will be constantly save throughout training",
                        type=str)
    parser.add_argument("--model_final_path",
                        help="relative path to where final model will be save",
                        type=str)
    parser.add_argument("--logging_path",
                        help="relative path to store logging",
                        type=str)
    parser.add_argument("--load_model_path",
                        help="relative path of the model to load",
                        type=str)
    parser.add_argument("--tokenizer_dataset_path",
                        help="path of the dataset for tokenizer to train on tokenizer ",
                        type=str)
    parser.add_argument("--tokenizer_path",
                        help="path for saving for tokenizer",
                        type=str)
    parser.add_argument("--vocab_size",
                        help="size of the tokenizer vocabulary",
                        type=int)
    parser.add_argument("--load_tokenizer_path",
                        help="relative path of the tokenizer to load",
                        type=bool)

    # tokenizer
    args = parser.parse_args()
    return args


# def getArgsModel(args):
#     final_args = {}
#     for attr in vars(args):
#         if attr in vars(model_train_config):
#             val = getattr(args, attr)
#             final_args[attr] = getattr(
#                 model_train_config, attr) if val is None else val
#         else:
#             final_args[attr] = None
#     return final_args


# def getArgsTokenizer(args):
#     final_args = {}
#     for attr in vars(args):
#         if attr in vars(tokenizer_config):
#             val = getattr(args, attr)
#             final_args[attr] = getattr(
#                 tokenizer_config, attr) if val is None else val
#         else:
#             final_args[attr] = None
#     return final_args

def getArgs(args, configuration):
    final_args = {}
    for attr in vars(args):
        if attr in vars(configuration):
            val = getattr(args, attr)
            final_args[attr] = getattr(
                configuration, attr) if val is None else val
        else:
            final_args[attr] = None
    return final_args


if __name__ == "__main__":
    args = createArgumentParser()
    assert args.model is not None, "Enter name of the mode using flag --model"
    if args.model == "t5_translation" or args.model == "t5_pretraining":
        final_args = getArgs(args, model_train_config)
    elif args.model == "sp":
        final_args = getArgs(args, tokenizer_config)
    else:
        raise ValueError(f"{args.model} is invalid.")
    trainer = run(**final_args)