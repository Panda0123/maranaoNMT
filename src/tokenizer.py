def trainSP(sourcePth: str,
            savePth: str,
            vocabSize: int,
            modelPrefix: str):
    import shutil
    import sentencepiece as sp
    print("Training Sentence Piece Tokenizer")
    command = ('--input=%s --model_prefix=%s --vocab_size=%s' %
               (sourcePth, modelPrefix, vocabSize))
    sp.SentencePieceTrainer.train(command)
    shutil.move(f"{modelPrefix}.vocab", savePth)
    shutil.move(f"{modelPrefix}.model", savePth)
    return sp
