from model.translation.mranax.transformer import Transformer
from model.translation.t5 import utils

from transformers import (
    T5Config, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    T5TokenizerFast
)

from functools import partial
import sys


def run(mrnDtPth: str, engDtPth: str,
        savePth: str, loggingPth: str):

    # load tokenizers
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    # mrnTok = T5TokenizerFast(mrnTokPth)

    bS = 16
    maxLen = 300
    # load Data
    # dataset = utils.getDataset(tokenizer, mrnDtPth, engDtPth, bS)
    dataset = utils.MyDataset(mrnDtPth, engDtPth)
    collateFn = partial(utils.collateFn, tokenizer=tokenizer)

    # load model
    modelConfig = T5Config(
        vocab_size=tokenizer.vocab_size,
        num_layers=3,
        dropout_rate=0.1,
        decoder_start_token_id=tokenizer.pad_token_id
    )
    model = T5ForConditionalGeneration(modelConfig)
    print(sum(p.numel() for p in model.parameters()))

    # instantiate training arguments
    args = Seq2SeqTrainingArguments(output_dir=savePth,
                                    do_train=True,
                                    overwrite_output_dir=True,
                                    # evaluation_strategy="epoch",
                                    per_device_train_batch_size=bS,
                                    learning_rate=1e-4,
                                    num_train_epochs=5,
                                    warmup_steps=500,
                                    seed=42,
                                    logging_dir=loggingPth,
                                    dataloader_num_workers=4)  # 4 * nGpu

    trainer = Seq2SeqTrainer(model=model,
                             args=args,
                             # max_length=maxLen,
                             # num_beams=4,
                             data_collator=collateFn,
                             train_dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    pass
    # import config
    # mrnDtPth = config.MRN_WOQ_CLEAN_PATH
    # engDtPth = config.ENG_WOQ_CLEAN_PATH
    # mrnTokPth = config.SP_PATH
    # savePth = config.T5_MODEL_PATH
    # loggingPth = config.T5_LOGGING_PATH
    # vocabSize = config.VOCAB_SIZE

    # run(mrnDtPth, engDtPth, mrnTokPth, savePth, loggingPth, vocabSize)
    # res = torch.cat((torch.tensor([engTok.pad_token_id]), trgIds[0]))
    # loss = model(input_ids=srcIds, labels=res))
