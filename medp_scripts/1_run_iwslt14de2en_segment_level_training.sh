#!/usr/bin/env bash

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20

# train
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --beam-search-hyps --segment-level-training-start-iter 6000 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion segment_level_training \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 3, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
