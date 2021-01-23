from model.translation.mranax.transformer import Transformer
from model.translation.mranax import utils

from transformers import T5TokenizerFast
from transformers import RobertaTokenizerFast
import torch as torch
import torch.nn as nn
import torch.optim as optim

from functools import partial
import time
import sys


def run(mrnDtPth: str, engDtPth: str,
        mrnTokPth: str, savePth: str,
        loggingPth: str):

    # load device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # load tokenizers
    engTok = RobertaTokenizerFast.from_pretrained("roberta-base")
    mrnTok = RobertaTokenizerFast.from_pretrained(mrnTokPth)

    # load Data
    bS = 16
    maxStep = 100
    trainLoader = utils.getDataset(mrnTok, engTok, mrnDtPth,
                                   engDtPth, bS, maxStep)

    # load model
    model = Transformer(
        maxStep=maxStep,
        embDim=512,
        trgVocabSize=engTok.vocab_size,
        srcVocabSize=mrnTok.vocab_size,
        nHeads=8,
        nDecoderLayers=3,
        nEncoderLayers=3,
        dropout=0.1,
        dimFeedForward=512*4,
        device=device).to(device)
    print("Model Num Parameters:",
          sum(p.numel() for p in model.parameters()))
    print("Model Num Trainable Parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # training parameters
    nEpochs = 2
    trgPadIdx = engTok.pad_token_id
    lossFn = nn.CrossEntropyLoss(ignore_index=trgPadIdx)
    opt = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, factor=0.1, verbose=True)
    logInterval = 100

    # training loop
    for epoch in range(nEpochs):
        model.train()
        runningLoss = 0.
        runningBatchTime = 0.
        epochLoss = 0.
        startTime = time.time()
        for i, batch in enumerate(trainLoader):
            # src|trg shape: [seqLen, bS]
            batchTime = time.time()
            src = batch['input_ids'].to(device)
            trg = batch['labels'].to(device)

            # outs shape: [seqLn, bS, trgVocabSize]
            outs = model(src, trg[:-1])
            # outs shape: [seqLn*bs, trgVocabSize]
            outs = outs.reshape(-1, outs.shape[-1])
            trg = trg[1:].reshape(-1)

            opt.zero_grad()
            loss = lossFn(outs, trg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            opt.step()

            runningLoss += loss.item()
            epochLoss += loss.item()
            runningBatchTime += time.time() - batchTime
            if i % logInterval == (logInterval - 1):
                print("[Epoch#%d %4d] Loss: %5.4f" %
                      (epoch+1, i+1, runningLoss/logInterval), end=" ")
                print("%4.2f ms/step" %
                      ((runningBatchTime * 1000) / logInterval))
                runningLoss = 0.
                runningBatchTime = 0.
        scheduler.step(epochLoss / (i + 1))
        print(f"ms/batch:{(time.time() - startTime) / (i + 1)}")


if __name__ == "__main__":
    import config
    mrnDtPth = config.MRN_WOQ_CLEAN_PATH
    engDtPth = config.ENG_WOQ_CLEAN_PATH
    mrnTokPth = config.BPE_PATH
    savePth = config.T5_MODEL_PATH
    loggingPth = config.T5_LOGGING_PATH
    vocabSize = config.VOCAB_SIZE
    run(mrnDtPth, engDtPth, mrnTokPth, savePth, loggingPth, vocabSize)
    engSen0 = "Are you youtuber?"
    engSen1 = "Are you okay?"
    tgtBtch = [engSen0, engSen1]
    mrnSen0 = "Baka youtuber?"
    mrnSen1 = "Mapipiya ka?"
    srcBtch = [mrnSen0, mrnSen1]

    engBtchTok = engTok(tgtBtch, padding='max_length', return_tensors="pt", max_length=100)
    engBtchTok = engTok(tgtBtch, padding=True, return_tensors="pt", max_length=100)
