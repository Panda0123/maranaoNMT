import argparse
import config
import tokenizer
from model.translation.t5.train import run as trainT5
from model.translation.mranax.train import run as mranax


def run(model: str, nEpochs: int):
    if model == "bpe":
        tokenizer.trainBPE(config.MRN_ALL_CLEAN_PATH,
                           config.BPE_PATH,
                           vocabSize=config.VOCAB_SIZE)
    if model == "sp":
        tokenizer.trainSP(config.MRN_ALL_CLEAN_PATH,
                          config.SP_PATH,
                          vocabSize=config.VOCAB_SIZE,
                          modelPrefix=model)
    if model == "mranax":
        mranax(config.MRN_WOQ_CLEAN_PATH,
               config.ENG_WOQ_CLEAN_PATH,
               config.BPE_PATH,
               config.MRANAX_MODEL_PATH,
               config.MRANAX_LOGGING_PATH)
    if model == "t5":
        trainer = trainT5(config.TRAIN_PATH,
                          config.VALIDATION_PATH,
                          config.T5_MODEL_PATH,
                          config.T5_MODEL_PATH_FINAL,
                          config.T5_LOGGING_PATH,
                          nEpochs, 16)

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--nepochs",
        type=int
    )
    args = parser.parse_args()
    model = args.model
    nEpochs = args.nepochs
    trainer = run(model, nEpochs)
