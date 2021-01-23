from transformers import T5TokenizerFast
from torch.utils.data import DataLoader, Dataset
from functools import partial


class MyDataset(Dataset):
    def __init__(self, mrnDtPth, engPth):
        super(MyDataset, self).__init__()
        self.data = []
        with open(mrnDtPth, "r") as mrnFl, open(engPth, "r") as engFl:
            for mrn, eng in zip(mrnFl, engFl):
                self.data.append(
                    {
                        "mrn": mrn.strip(),
                        "eng": eng.strip()
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collateFn(data: list, mrnTok, engTok, maxStep):
    mrns = []
    engs = []
    for instance in data:
        mrns.append(instance['mrn'])
        engs.append(instance['eng'])
    mrns = mrnTok(mrns, padding='max_length',
                  max_length=maxStep, return_tensors="pt")
    engs = engTok(engs, padding='max_length',
                  max_length=maxStep, return_tensors="pt")

    return {
            "input_ids": mrns.input_ids.transpose(0, 1),
            "attention_mask": mrns.attention_mask,
            "labels": engs.input_ids.transpose(0, 1)
           }


def getDataset(mrnTok, engTok, mrnDtPth, engDtPth, bS, maxStep):
    dataset = MyDataset(mrnDtPth, engDtPth)
    collate = partial(collateFn, mrnTok=mrnTok,
                      engTok=engTok, maxStep=maxStep)
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
