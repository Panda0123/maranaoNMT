from transformers import T5TokenizerFast
from torch.utils.data import DataLoader, Dataset
import numpy as np

from functools import partial
import os
import math


def collateFn(data: list, tokenizer):
    src = []
    labels = []
    for instance in data:
        src.append(instance[0])
        labels.append(instance[1])

    inputs = tokenizer(src, padding="longest", return_tensors="pt")
    outputs = tokenizer(labels, padding="longest", return_tensors="pt")

    return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": outputs.input_ids,
            "decoder_attention_mask": outputs.attention_mask
           }

def checkDir(pth: str):
    # check if dir exists
    assert os.path.isdir(pth), f"{pth} is not a directory or does not exist."

def checkFile(pth: str):
    # check if file exists
    assert os.path.isfile(pth), f"{pth} is not a file or does not exists."
    
# === Dataset for Translation ===
class TranslationDataset(Dataset):
    def __init__(self, path):
        super(TranslationDataset, self).__init__()
        self.data = []
        mrnPth = path + "maranao.txt"
        engPth = path + "english.txt"

        assert os.path.isfile(mrnPth), f"{mrnPth} does not exist"
        assert os.path.isfile(engPth), f"{engPth} does not exist"

        with open(mrnPth, "r") as mrnFl, open(engPth, "r") as engFl:
            for mrn, eng in zip(mrnFl, engFl):
                src = "translate Maranao to English: " + mrn
                self.data.append((src, eng))
                src = "translate English to Maranao: " + eng
                self.data.append((src, mrn))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]





def getDataset(tokenizer, mrnDtPth, engDtPth, bS):
    dataset = MyDataset(mrnDtPth, engDtPth)
    collate = partial(collateFn, tokenizer=tokenizer)
    dataLoader = DataLoader(dataset, batch_size=bS,
                            shuffle=True, collate_fn=collate)
    return dataLoader


def getRawDataset(mrnDtPth, engPth):
    data = []
    with open(mrnDtPth, "r") as mrnFl, open(engPth, "r") as engFl:
        for mrn, eng in zip(mrnFl, engFl):
            data.append(
                {
                    "mrn": mrn.strip(),
                    "eng": eng.strip()
                }
            )
    return data


# === Dataset and Preprocessing for Pretraining ===
def span_corruption(inputs, tokenizer, noiseDensity=0.15, spanLen=3):
    """
        Preprocessor for pretraining by injecting noise for 1 instance.
        args:
            inputs: list - ids of the tokens. including the EOS token.
            tokenizer: transformers.T5Tokenizer - pretrained tokenizer from hugging face.
            noiseDensity: float -  percentange of tokens to be corrupted.
            spanLen: int - maximum length of span. if inputs length is lower than this value no noise added.
        returns:
            outs: dictionary - a dictionary containing the inputs and labels
                e.g.
                    {
                        "inputs": [123, 34532, 12, 4, 3],
                        "labels": [34534, 42, 231, 42,  34533]
                    }
    """
    specialTokens = tokenizer.convert_tokens_to_ids([f"<extra_id_{idx}>" for idx in range(0, 100)])  # special tokens
    nTokens = len(inputs[:-1]) # exclude eos token
    nCorTokens = int(round(nTokens * noiseDensity))
    nSpan = int(math.ceil(nCorTokens / spanLen)) if nCorTokens >= spanLen or nCorTokens == 0 else 1  # number of span
    tokensRem =  nCorTokens
    selectedIdx = []  # corrupted span
    for _ in range(nSpan):
        isIn = True
        spanLen = spanLen if tokensRem >= spanLen else tokensRem
        while isIn:
            idx = np.random.randint(0, nTokens-spanLen)
            isIn = any([idx in idxs for idxs in selectedIdx])
        selectedIdx.append([i for i in range(idx, idx+spanLen)])
        tokensRem -= spanLen

    selectedIdx = sorted(selectedIdx, key=lambda x: x[0])
    inps = []
    labels = []
    lblsOffset = 0

    # add corrupt  append instead of del
    for i, span in enumerate(selectedIdx):
        strIdx = span[0]
        endIdx = span[-1]
        if i == 0:
            if strIdx == 0:
                inps.append(specialTokens[i])
                labels += inputs[strIdx:endIdx] if strIdx != endIdx else [inputs[strIdx]]
                lblsOffset += 1
            else:
                inps += inputs[0:strIdx]
                inps.append(specialTokens[i])
                labels.append(specialTokens[i])
                labels += inputs[strIdx:endIdx] if strIdx != endIdx else [inputs[strIdx]]
        else:
            if strIdx - selectedIdx[i-1][-1] > 0:
                inps += inputs[selectedIdx[i-1][-1]: strIdx]
                inps.append(specialTokens[i])
                labels.append(specialTokens[i - lblsOffset])
                labels += inputs[strIdx:endIdx] if strIdx != endIdx else [inputs[strIdx]]

    if len(selectedIdx) != 0:
        # check if last span is the end of the sequence
        if endIdx != nTokens:
            inps += inputs[endIdx:nTokens] if strIdx != endIdx else inputs[endIdx+1:nTokens]
            inps.append(inputs[-1])  # add EOS to labels
            labels.append(specialTokens[i+1 - lblsOffset])
            labels.append(inputs[-1])  # add EOS to labels
    else:
        # same labels and inputs
        inps = inputs
        labels = inputs

    return {
        "inputs": inps,
        "labels": labels
    }

    
class PretrainingDataset(Dataset):

    def __init__(self, path, tokenizer):
        super(PretrainingDataset, self).__init__()
        self.data = []
        assert os.path.isfile(path), f"{path} does not exist"

        # TODO: add multithreading for loading data
        with open(path, "r", errors="ignore") as fl:
            for line in fl.readlines():
                dct = span_corruption(tokenizer(line).input_ids,
                                      tokenizer,
                                      noiseDensity=0.15,
                                      spanLen=3)
                inputs, labels = tokenizer.decode(dct["inputs"])[:-4], tokenizer.decode(dct["labels"])[:-4]
                self.data.append((inputs, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]