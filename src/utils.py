from sklearn.cluster import KMeans
import numpy as np
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
import os 
import torch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertTokenizer
import numpy as np
import os
import string
from nltk.tag import pos_tag


def cluster_eval(label_path, emb_path):
    labels = open(label_path).readlines()
    labels = np.array([int(label.strip()) for label in labels])
    n_clusters = len(set(labels))
    embs = torch.load(emb_path)
    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(embs.numpy())
    nmi = normalized_mutual_info_score(y_pred, labels)
    print(f"NMI score: {nmi:.4f}")
    return nmi

def split_doc(docs, max_len):
    new_docs = []
    new_doc = []
    for doc in docs:
        sents = sent_tokenize(doc)
        if len(new_doc) > 0:
            new_docs.append(' '.join(new_doc))
        new_doc = []
        new_doc_len = 0
        for sent in sents:
            words = word_tokenize(sent)
            if new_doc_len + len(words) > max_len:
                new_docs.append(' '.join(new_doc))
                new_doc = [sent]
                new_doc_len = len(words)
            else:
                new_doc.append(sent)
                new_doc_len += len(words)
    return new_docs

def corpus_trunc_stats(docs, max_len):
    doc_len = []
    for doc in docs:
        input_ids = tokenizer.encode(doc, add_special_tokens=True)
        doc_len.append(len(input_ids))
    print(f"Document max length: {np.max(doc_len)}, avg length: {np.mean(doc_len)}, std length: {np.std(doc_len)}")
    trunc_frac = np.sum(np.array(doc_len) > max_len) / len(doc_len)
    print(f"Truncated fraction of all documents: {trunc_frac}")

def encode(docs, max_len=512):
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=max_len, padding='max_length',
                                                    return_attention_mask=True, truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

def create_dataset(dataset_dir, text_file, loader_name, max_len=512):
    loader_file = os.path.join(dataset_dir, loader_name)
    if os.path.exists(loader_file):
        print(f"Loading encoded texts from {loader_file}")
        data = torch.load(loader_file)
    else:
        print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
        corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
        docs = []
        for doc in corpus.readlines():
            content = doc.strip().split('\t')
            assert len(content) == 2
            docs.append(content[-1])
        print(f"Converting texts into tensors.")
        input_ids, attention_masks = encode(docs, max_len)
        print(f"Saving encoded texts into {loader_file}")
        stop_words = set(stopwords.words('english'))
        filter_idx = []
        valid_pos = ["NOUN", "VERB", "ADJ"]
        for i in inv_vocab:
            token = inv_vocab[i]
            if token in stop_words or token.startswith('##') \
               or token in string.punctuation or token.startswith('[') \
               or pos_tag([token], tagset='universal')[0][-1] not in valid_pos:
                filter_idx.append(i)
        print(f"valid vocab: {len(vocab) - len(filter_idx)}")
        valid_pos = attention_masks.clone()
        for i in filter_idx:
            valid_pos[input_ids == i] = 0
        data = {"input_ids": input_ids, "attention_masks": attention_masks, "valid_pos": valid_pos}
        torch.save(data, loader_file)
    return data

pretrained_lm = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
vocab = tokenizer.get_vocab()
vocab_size = len(vocab)
inv_vocab = {k:v for v, k in vocab.items()}
