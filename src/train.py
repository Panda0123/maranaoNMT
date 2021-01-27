import argparse
import sys

import config
import tokenizer
from model.translation.t5.train import run as trainT5


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
    logging_path: str
):
    trainer = None

    if model == "sp":
        trainer = tokenizer.trainSP(config.TOKENIZER_DATASET_PATH,
                                    config.SP_PATH,
                                    vocabSize=config.VOCAB_SIZE,
                                    modelPrefix=model)
    if model == "t5":
        trainer = trainT5(num_layers,
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
                          logging_path)

    assert trainer is not None, f"{model} is invalid."
    return trainer


def createArgumentParser():
    parser = argparse.ArgumentParser(prog="Trainer")
    # model
    parser.add_argument("--model",
                        help="name of the model default to t5",
                        default=config.model,
                        type=str)
    parser.add_argument("--num_layers",
                        help="number of layers for both encoder and decoder",
                        default=config.num_layers,
                        type=int)
    parser.add_argument("--dropout_rate",
                        default=config.dropout_rate,
                        type=float)

    # training
    parser.add_argument("--num_epochs",
                        default=config.num_epochs,
                        type=int)
    parser.add_argument("--batch_size",
                        default=config.batch_size,
                        type=int)
    parser.add_argument("--lr",
                        default=config.lr,
                        type=float)
    parser.add_argument("--warmup_steps",
                        default=config.warmup_steps,
                        type=int)
    parser.add_argument("--num_workers",
                        help="number of dataloader workers",
                        default=config.num_workers,
                        type=int)
    parser.add_argument("--save_interval",
                        help="number of steps between two save checkpoint",
                        default=config.save_interval,
                        type=int)
    parser.add_argument("--log_interval",
                        help="number of steps between two logs",
                        default=config.log_interval,
                        type=int)

    # paths
    parser.add_argument("--train_path",
                        help="relative path to the train dataset",
                        default=config.train_path,
                        type=str)
    parser.add_argument("--validation_path",
                        help="relative path to the valid dataset",
                        default=config.validation_path,
                        type=str)
    parser.add_argument("--model_path",
                        help="relative path to where model will be constantly save",
                        default=config.model_path,
                        type=str)
    parser.add_argument("--model_final_path",
                        help="relative path to where final model will be save",
                        default=config.model_final_path,
                        type=str)
    parser.add_argument("--logging_path",
                        help="relative path to store logging",
                        default=config.logging_path,
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = createArgumentParser()
    trainer = run(**vars(args))
