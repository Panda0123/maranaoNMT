def trainSP(sourcePth: str, savePth: str, vocabSize: int, modelPrefix: str):
    import sentencepiece as sp
    command = ('--input=%s --model_prefix=%s --vocab_size=%s' %
               (sourcePth, modelPrefix, vocabSize))
    sp.SentencePieceTrainer.train(command)
