# Pre-Training imputation

from torch.data.util import DataLoader

from .utils import MyDataset

trainPth = "../../../data/mrnWOQClnd.txt"
# TODO: cange path for valid
validPth = "../../../data/mrnWOQClnd.txt"

trainDts = MyDataset(trainPth)
validDts = MyDataset(validPth)

trainLoader = DataLoader(trainDts,)
