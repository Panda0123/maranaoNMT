def trainSP(sourcePth: str,
            savePth: str,
            vocabSize: int,
            modelPrefix: str):
    import shutil
    import sentencepiece as sp
    print("Training Sentence Piece Tokenizer")
    # special tokens ids are modified to fit with t5 tokenizer from hugging face
    command = ('--input=%s --model_prefix=%s --bos_id=-1 --pad_id=0 --eos_id=1 --unk_id=2 --vocab_size=%s' %
               (sourcePth, modelPrefix, vocabSize))
    sp.SentencePieceTrainer.train(command)
    shutil.move(f"{modelPrefix}.vocab", savePth)
    shutil.move(f"{modelPrefix}.model", savePth)
    return sp