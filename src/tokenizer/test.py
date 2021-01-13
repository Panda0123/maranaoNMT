from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer(
    "../../data/tokenizer/tokenizer-vocab.json",
    "../../data/tokenizer/tokenizer-merges.txt"
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("<eos>", tokenizer.token_to_id("<eos>")),
    ("<sos>", tokenizer.token_to_id("<sos>"))
)
tokenizer.enable_truncation(max_length=100)
sentence = "Miyakaisa ko makaaloyan ko walay ran."
res = tokenizer.encode(sentence)
print(res.tokens)
