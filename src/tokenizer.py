
def trainSP(sourcePth: str, savePth: str, vocabSize: int, modelPrefix: str):
    import sentencepiece as sp
    command = ('--input=%s --model_prefix=%s --vocab_size=%s' %
               (sourcePth, modelPrefix, vocabSize))
    sp.SentencePieceTrainer.train(command)


def trainBPE(sourcePth: str, savePth: str, vocabSize: int):
    # traing bytelevell byte pair encoding tokenizer
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train([sourcePth], vocab_size=vocabSize, min_frequency=2,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.save_model(savePth)