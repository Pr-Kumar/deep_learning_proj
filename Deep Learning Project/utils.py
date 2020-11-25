import re
import nltk
import numpy as np
from collections import Counter

def pad_input(sentences, seq_len):
    # Pads/Trims sequences to common length
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def preprocessData(train_file_lines, test_file_lines, num_train, num_test, seq_len):
    # Decode files
    train_file = [x.decode('utf-8') for x in train_file_lines[:num_train]]
    test_file = [x.decode('utf-8') for x in test_file_lines[:num_test]]

    # Extracting labels from sentences
    train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
    train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]
    test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]
    test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]

    # Some simple cleaning of data
    # Modify URLs to <url>
    for i in range(len(train_sentences)):
        train_sentences[i] = re.sub('\d','0',train_sentences[i])
        if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
            train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])
    for i in range(len(test_sentences)):
        test_sentences[i] = re.sub('\d','0',test_sentences[i])
        if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
            test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

    # Print one sentence and label as sanity check
    # print(train_sentences[0])
    # print(train_labels[0])

    # Create Counter to map word to count
    words = Counter()
    for i, sentence in enumerate(train_sentences):
        train_sentences[i] = []
        for word in nltk.word_tokenize(sentence): # Tokenizing the words
            words.update([word.lower()]) # Converting all the words to lower case
            train_sentences[i].append(word)
        if i%20000 == 0:
            print(str((i*100)/num_train) + "% done")
    print("100% done")
    # Removing the words that only appear once, sort by appearance and add padding
    words = {k:v for k,v in words.items() if v>1}
    words = sorted(words, key=words.get, reverse=True)
    words = ['_PAD','_UNK'] + words

    # Dictionaries to store the word to index mappings and vice versa
    word2idx = {o:i for i,o in enumerate(words)}
    idx2word = {i:o for i,o in enumerate(words)}

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return train_sentences, test_sentences, train_labels, test_labels, word2idx, idx2word

def normalizeSentences(train_sentences, test_sentences, word2idx, seq_len):
    # Tokenize sentences
    for i, sentence in enumerate(train_sentences):
      train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]
    for i, sentence in enumerate(test_sentences):
      test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

    # Print converted sentence as sanity check
    # print(len(train_sentences[0]))
    # print(train_sentences[0])

    # Pad inputs to conform to common sequence length
    train_sentences = pad_input(train_sentences, seq_len)
    test_sentences = pad_input(test_sentences, seq_len)

    return train_sentences, test_sentences