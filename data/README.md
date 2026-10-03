# Pinned data

These files make the experiments runnable with no network access. Every load through `pinned_data.py` checks the SHA-256 fingerprints in `MANIFEST.json`.

## GPT-2 tokenizer: `gpt2/`

`vocab.bpe` and `encoder.json` are OpenAI's GPT-2 byte-pair encoding files. Their fingerprints match the ones the `tiktoken` library pins for its `gpt2` encoding, so they are byte-identical to the files `tiktoken` downloads. This copy was taken from the `gpt-3-encoder` npm package, version 1.1.4.

OpenAI released these files with GPT-2 under the modified MIT license reproduced in `gpt2/LICENSE`, taken from the `openai/gpt-2` repository.

## WikiText-2 raw: `wikitext-2-raw-v1/`

Status: pending. Running `scripts/fetch_wikitext2.py` once downloads the Hugging Face copy and pins one gzip-compressed text file per split. Each line of a file is one row of the dataset.

The script stops unless the copy reproduces the original run exactly:

| Check | Expected |
|---|---|
| Rows in train / validation / test | 36,718 / 3,760 / 4,358 |
| GPT-2 tokens in train, joined as in the notebook | 2,391,884 |

These values were printed by the original Colab run of the full experiments notebook, in git commit `6b65d64`.

WikiText was created by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher from Wikipedia articles. It is distributed under the Creative Commons Attribution-ShareAlike 3.0 license, so the files in `wikitext-2-raw-v1/` stay under that license rather than the repository's MIT license.

> Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer sentinel mixture models. arXiv:1609.07843.
