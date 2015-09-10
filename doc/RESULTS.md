## One Billion Word Benchmark

### Corpus description
Vocabulary size: 793470 words

Training dataset:
 - words: 768646533
 - md5 hash: 3df74e73b531faf23e4846e559c2d466

Validation dataset (heldout-00000):
 - words: 153583
 - md5 hash: ee2b6b23070db794a39b42629187bc13

Test dataset (heldout-00001):
 - words: 159354
 - md5 hash: d6f3176e6323e1fb3cc24ad231a91f5a

All OOV words are explicitly replaced with "<unk>" in all the datasets.

### LM Results

```
Args: --hidden 256 --hidden-type gru-insyn --nce 0 --direct-order 0 --direct 0 --alpha 0.05 --bptt 32 --bptt-skip 8
Validation entropy: 7.520659
Test entropy: 7.520659
```
```
Args: --hidden 256 --hidden-type gru-insyn --nce 50 --alpha 0.05 --bptt-skip 8 --bptt 32
Validation entropy: 6.944975
Test entropy: 6.944975
```
```
Args: --hidden 256 --hidden-type gru-insyn --nce 50 --direct-order 4 --direct 1000 --alpha 0.05 --bptt-skip 8 --bptt 32
Validation entropy: 6.426476
Test entropy: 6.426476
```

```
Args: --hidden 256 --hidden-type relu-trunc --nce 0 --direct-order 0 --direct 0 --diagonal-initialization 0.9 --alpha 0.01 --bptt 1 --bptt-skip 8
Validation entropy: 7.606524
Test entropy: 7.606524
```
```
Args: --hidden 256 --hidden-type relu-trunc --nce 50 --diagonal-initialization 0.1 --alpha 0.03 --bptt-skip 8 --bptt 3
Validation entropy: 7.487164
Test entropy: 7.487164
```

```
Args: --hidden 256 --hidden-type scrn40 --nce 0 --direct-order 0 --direct 0 --alpha 0.1 --bptt 32 --bptt-skip 8
Validation entropy: 7.486989
Test entropy: 7.486989
```
```
Args: --hidden 256 --hidden-type scrn40 --nce 50 --alpha 0.1 --bptt-skip 8 --bptt 32
Validation entropy: 7.037301
Test entropy: 7.037301
```
```
Args: --hidden 256 --hidden-type scrn40 --nce 50 --direct-order 4 --direct 1000 --alpha 0.1 --bptt-skip 8 --bptt 32
Validation entropy: 6.464881
Test entropy: 6.464881
```

```
Args: --hidden 256 --hidden-type sigmoid --nce 0 --direct-order 0 --direct 0 --alpha 0.1 --bptt 50 --bptt-skip 1
Validation entropy: 7.373024
Test entropy: 7.373024
```
```
Args: --hidden 256 --hidden-type sigmoid --nce 50 --alpha 0.1 --bptt-skip 1 --bptt 50
Validation entropy: 6.893834
Test entropy: 6.893834
```
```
Args: --hidden 256 --hidden-type sigmoid --nce 50 --direct-order 4 --direct 1000 --alpha 0.1 --bptt-skip 1 --bptt 50
Validation entropy: 6.457476
Test entropy: 6.457476
```

```
Args: --hidden 256 --hidden-type tanh --nce 0 --direct-order 0 --direct 0 --alpha 0.1 --bptt 50 --bptt-skip 1
Validation entropy: 7.477661
Test entropy: 7.477661
```
```
Args: --hidden 256 --hidden-type tanh --nce 50 --alpha 0.1 --bptt-skip 1 --bptt 50
Validation entropy: 6.978361
Test entropy: 6.978361
```
```
Args: --hidden 256 --hidden-type tanh --nce 50 --direct-order 4 --direct 1000 --alpha 0.1 --bptt-skip 1 --bptt 50
Validation entropy: 6.65744
Test entropy: 6.65744
```
