from transformers import (
    T5Config, T5ForConditionalGeneration,
    T5TokenizerFast
)

import config

model = T5ForConditionalGeneration.from_pretrained(config.T5_MODEL_PATH_FINAL)
tokenizer = T5TokenizerFast.from_pretrained("t5-small")


mrn = "She feared the thunder."
src = "translate English to Maranao: " + mrn
tok = tokenizer(src, return_tensors="pt").input_ids
res = model.generate(tok)
print(tokenizer.decode(res[0]))
