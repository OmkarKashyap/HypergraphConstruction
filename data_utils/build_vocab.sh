#!/bin/bash
# build vocab for different datasets

python data_utils/prepare_vocab.py --data_dir dataset/Restaurants_corenlp --vocab_dir dataset/Restaurants_corenlp
python data_utils/prepare_vocab.py --data_dir dataset/Laptops_corenlp --vocab_dir dataset/Laptops_corenlp
python data_utils/prepare_vocab.py --data_dir dataset/Tweets_corenlp --vocab_dir dataset/Tweets_corenlp

python data_utils/prepare_vocab.py --data_dir dataset/Restaurants_allennlp --vocab_dir dataset/Restaurants_allennlp
python data_utils/prepare_vocab.py --data_dir dataset/Laptops_allennlp --vocab_dir dataset/Laptops_allennlp
python data_utils/prepare_vocab.py --data_dir dataset/Tweets_allennlp --vocab_dir dataset/Tweets_allennlp

python data_utils/prepare_vocab.py --data_dir dataset/Restaurants_stanza --vocab_dir dataset/Restaurants_stanza
python data_utils/prepare_vocab.py --data_dir dataset/Laptops_stanza --vocab_dir dataset/Laptops_stanza
python data_utils/prepare_vocab.py --data_dir dataset/Tweets_stanza --vocab_dir dataset/Tweets_stanza