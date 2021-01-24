from transformers import T5TokenizerFast
from torch.utils.data import DataLoader, Dataset

from functools import partial
import os


class MyDataset(Dataset):
    def __init__(self, path):
        super(MyDataset, self).__init__()
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
