from model.translation.mranax.transformer import Transformer
from model.translation.t5 import utils

from transformers import (
    T5Config, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    T5TokenizerFast
)

from functools import partial
import sys
import os


def checkPath(pth):
    # check if dir exists
    assert os.path.isdir(pth), f"{pth} is not a dir."


def run(
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
    load_model: bool,
):

    # check paths
    checkPath(train_path)
    checkPath(validation_path)
    checkPath(model_path)
    checkPath(model_final_path)
    checkPath(logging_path)

    # load tokenizers
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    # load Data
    trainDts = utils.MyDataset(train_path)
    validDts = utils.MyDataset(validation_path)
    collateFn = partial(utils.collateFn, tokenizer=tokenizer)

    # load model
    if load_model:
        print(f"Load model in {model_final_path}")
        model = T5ForConditionalGeneration.from_pretrained(model_final_path)
        model.train()
    else:
        modelConfig = T5Config(
            vocab_size=tokenizer.vocab_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            decoder_start_token_id=tokenizer.pad_token_id
        )
        model = T5ForConditionalGeneration(modelConfig)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    # instantiate training arguments
    args = Seq2SeqTrainingArguments(output_dir=model_path,
                                    overwrite_output_dir=True,
                                    do_train=True,
                                    do_eval=True,
                                    evaluation_strategy="epoch",
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    learning_rate=lr,
                                    num_train_epochs=num_epochs,
                                    warmup_steps=warmup_steps,
                                    seed=42,
                                    logging_dir=logging_path,
                                    load_best_model_at_end=False,
                                    save_total_limit=2,
                                    save_steps=save_interval,
                                    logging_steps=log_interval,
                                    dataloader_num_workers=num_workers)

    trainer = Seq2SeqTrainer(model=model,
                             args=args,
                             # max_length=maxLen,
                             # num_beams=4,
                             data_collator=collateFn,
                             train_dataset=trainDts,
                             eval_dataset=validDts)
    trainer.train()
    trainer.save_model(model_final_path)
    return trainer
