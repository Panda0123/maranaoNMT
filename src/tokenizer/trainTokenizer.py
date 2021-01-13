# traing bytelevell byte pair encoding tokenizer
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

path = "../../data/mrnAllClnd.txt"
tokenizer.train([path], vocab_size=30000, min_frequency=2,
        special_tokens=["<sos>", "<pad>", "<eos>", "<unk>", "<mask>"])

tokenizer.save_model("../../data/tokenizer/", "tokenizer")
