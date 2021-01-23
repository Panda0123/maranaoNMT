import torch
import torch.nn as nn
import numpy as np


# MODEL CREATION
class PositionalEncoding(nn.Module):
    def __init__(self, maxStep, embDim, device):
        super(PositionalEncoding, self).__init__()
        encoding = torch.empty(maxStep, embDim).to(device)

        for i in range(embDim // 2):
            encoding[:, i*2] = torch.tensor([np.sin(
                p / (10000 ** (2 * i / embDim))) for p in range(maxStep)])
            encoding[:, i*2 + 1] = torch.tensor([np.cos(
                p / (10000 ** (2 * i / embDim))) for p in range(maxStep)])
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        return self.encoding[:x.size(-2), :x.size(-1)]


class Transformer(nn.Module):
    def __init__(self,
                 maxStep,
                 embDim,
                 trgVocabSize,
                 srcVocabSize,
                 nHeads,
                 nDecoderLayers,
                 nEncoderLayers,
                 dropout,
                 dimFeedForward,
                 device):
        super(Transformer, self).__init__()
        self.device = device
        self.srcEmb = nn.Embedding(srcVocabSize, embDim)
        self.srcPosEnc = PositionalEncoding(maxStep, embDim, device)
        self.trgEmb = nn.Embedding(trgVocabSize, embDim)
        self.trgPosEnc = PositionalEncoding(maxStep, embDim, device)
        self.transformer = nn.Transformer(
            embDim, nHeads, nEncoderLayers,
            nDecoderLayers, dimFeedForward, dropout)
        self.fc = nn.Linear(embDim, trgVocabSize)
        self.dropoutLayer = nn.Dropout(dropout)

    def forward(self, src, trg):
        # src shape: [maxStep, N]
        # trg shape: [maxStep, N]
        srcEmbedded = self.srcEmb(src)
        srcEmbedded = self.dropoutLayer(
            srcEmbedded + self.srcPosEnc(srcEmbedded))
        trgEmbedded = self.trgEmb(trg)
        trgEmbedded = self.dropoutLayer(
            trgEmbedded + self.trgPosEnc(trgEmbedded))

        trgMask = self.transformer.generate_square_subsequent_mask(
            trg.size(0)).to(self.device)
        outs = self.transformer(srcEmbedded, trgEmbedded, tgt_mask=trgMask)
        return self.fc(outs)
