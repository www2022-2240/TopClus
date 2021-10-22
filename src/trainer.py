from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from nltk.corpus import stopwords
import string
from nltk.tag import pos_tag
from transformers import BertTokenizer
from model import TopClusModel
import os
from tqdm import tqdm
import argparse
from sklearn.cluster import KMeans
from utils import cluster_eval
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


class TopClusTrainer(object):

    def __init__(self, args):
        self.args = args
        pretrained_lm = 'bert-base-uncased'
        self.n_clusters = args.n_clusters
        self.model = TopClusModel.from_pretrained(pretrained_lm,
                                                  output_attentions=False,
                                                  output_hidden_states=False,
                                                  input_dim=args.input_dim,
                                                  hidden_dims=eval(args.hidden_dims),
                                                  n_clusters=args.n_clusters,
                                                  temp=args.temperature)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = eval(args.hidden_dims)[-1]
        tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
        self.vocab = tokenizer.get_vocab()
        self.inv_vocab = {k:v for v, k in self.vocab.items()}
        self.filter_vocab()
        self.data_dir = os.path.join("datasets", args.dataset)
        data = self.load_dataset(self.data_dir, "text.pt")
        input_ids = data["input_ids"]
        attention_masks = data["attention_masks"]
        valid_pos = data["valid_pos"]
        doc_ids = torch.arange(input_ids.size(0))
        self.data = TensorDataset(doc_ids, input_ids, attention_masks, valid_pos)
        self.batch_size = args.batch_size
        self.update_interval = args.update_interval
        self.temp_dir = f"tmp_{args.seed}"
        self.aspects = ["food", "sent"] if args.dataset == "yelp" else ["topic", "location"]
        os.makedirs(self.temp_dir, exist_ok=True)
        self.log_files = {}
        for aspect in self.aspects:
            self.log_files[aspect] = os.path.join(self.temp_dir, f"log_{aspect}.txt")
            f = open(self.log_files[aspect], 'w')
            f.close()

    def load_dataset(self, dataset_dir, loader_name):
        loader_file = os.path.join(dataset_dir, loader_name)
        assert os.path.exists(loader_file)
        print(f"Loading encoded texts from {loader_file}")
        data = torch.load(loader_file)
        return data

    def pretrain(self, pretrain_epoch=10):
        sampler = RandomSampler(self.data)
        dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
        model = self.model.to(self.device)
        model.eval()
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr)
        for epoch in range(pretrain_epoch):
            total_loss = 0
            for batch_idx, batch in enumerate(tqdm(dataset_loader)):
                optimizer.zero_grad()
                input_ids = batch[1].to(self.device)
                attention_mask = batch[2].to(self.device)
                input_embs, output_embs = model.pretrain_forward(input_ids, attention_mask)
                loss = F.mse_loss(output_embs, input_embs)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"epoch {epoch}: loss = {total_loss / (batch_idx+1):.4f}")
        pretrained_path = os.path.join(self.data_dir, "pretrained.pt")
        torch.save(model.ae.state_dict(), pretrained_path)
        print(f"model saved to {pretrained_path}")

    def cluster_init(self):
        latent_emb_path = os.path.join(self.data_dir, "init_latent_emb.pt")
        model = self.model.to(self.device)
        if os.path.exists(latent_emb_path) and os.path.exists(latent_emb_path):
            print(f"Loading initial latent embeddings from {latent_emb_path}")
            latent_embs, freq = torch.load(latent_emb_path)
        else:
            sampler = SequentialSampler(self.data)
            dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
            model.eval()
            latent_embs = torch.zeros((len(self.vocab), self.latent_dim)).to(self.device)
            freq = torch.zeros(len(self.vocab), dtype=int).to(self.device)
            with torch.no_grad():
                for batch in tqdm(dataset_loader, desc="Obtaining initial latent embeddings"):
                    input_ids = batch[1].to(self.device)
                    attention_mask = batch[2].to(self.device)
                    valid_pos = batch[3].to(self.device)
                    latent_emb = model.init_emb(input_ids, attention_mask, valid_pos)
                    valid_ids = input_ids[valid_pos != 0]
                    latent_embs.index_add_(0, valid_ids, latent_emb)
                    freq.index_add_(0, valid_ids, torch.ones_like(valid_ids))
            latent_embs = latent_embs[freq > 0].cpu()
            freq = freq[freq > 0].cpu()
            latent_embs = latent_embs / freq.unsqueeze(-1)
            print(f"Saving initial embeddings to {latent_emb_path}")
            torch.save((latent_embs, freq), latent_emb_path)

        print(f"Running K-Means for initialization")
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(latent_embs.numpy(), sample_weight=freq.numpy())
        model.topic_emb.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

    def filter_vocab(self):
        stop_words = set(stopwords.words('english'))
        self.filter_idx = []
        for i in self.inv_vocab:
            token = self.inv_vocab[i]
            if token in stop_words or token.startswith('##') \
               or token in string.punctuation or token.startswith('['):
                self.filter_idx.append(i)

    def inference(self, topk=10, suffix=""):
        sampler = SequentialSampler(self.data)
        dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
        model = self.model.to(self.device)
        model.eval()
        latent_doc_embs = []
        doc_embs = []
        preds = []
        word_topic_sim_mean = -1 * torch.ones((len(self.vocab), self.n_clusters))
        word_topic_sim_dict = defaultdict(list)
        with torch.no_grad():
            for batch in tqdm(dataset_loader, desc="Inference"):
                input_ids = batch[1].to(self.device)
                attention_mask = batch[2].to(self.device)
                doc_emb, z, p, word_ids, sim = model.inference(input_ids, attention_mask)
                doc_embs.append(doc_emb.detach().cpu())
                latent_doc_embs.append(z.detach().cpu())
                preds.append(p.detach())
                for word_id, s in zip(word_ids, sim):
                    word_topic_sim_dict[word_id.item()].append(s.cpu().unsqueeze(0))
        for i in range(len(word_topic_sim_mean)):
            if len(word_topic_sim_dict[i]) > 5:
                word_topic_sim_mean[i] = torch.cat(word_topic_sim_dict[i], dim=0).mean(dim=0)
        word_topic_sim_mean[self.filter_idx, :] = -1
        topic_sim_mat = torch.matmul(model.topic_emb, model.topic_emb.t())
        start_idx = topic_sim_mat.sum(-1).argmax()
        sort_idx = topic_sim_mat[start_idx].argsort(descending=True).cpu().numpy()
        topic_file = open(os.path.join(self.temp_dir, f"mean_topics{suffix}.txt"), "w")
        for j in sort_idx:
            top_val, top_idx = torch.topk(word_topic_sim_mean[:, j], topk)
            result_string = []
            for val, idx in zip(top_val, top_idx):
                result_string.append(f"{self.inv_vocab[idx.item()]}")
            topic_file.write(f"Topic {j}: {','.join(result_string)}\n")

        doc_embs = torch.cat(doc_embs, dim=0)
        latent_doc_embs = torch.cat(latent_doc_embs, dim=0)
        preds = torch.cat(preds, dim=0)

        doc_emb_path = os.path.join(self.temp_dir, "latent_doc_emb.pt")
        print(f"Saving document embeddings to {doc_emb_path}")
        torch.save(latent_doc_embs, doc_emb_path)

        for aspect in self.aspects:
            print(f"Evaluating latent embeddings on {aspect}")
            label_path = os.path.join(self.data_dir, f"label_{aspect}.txt")
            nmi = cluster_eval(label_path, doc_emb_path)
            f = open(self.log_files[aspect], 'a')
            f.write(f"{suffix.split('_')[-1]}:\t{nmi:.2f}\n")

        return preds

    def target_distribution(self, preds):
        targets = preds**2 / preds.sum(dim=0)
        targets = (targets.t() / targets.sum(dim=1)).t()
        return targets

    def clustering(self, epochs=20):
        self.cluster_init()
        sampler = RandomSampler(self.data)
        dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
        model = self.model.to(self.device)
        model.eval()
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr)
        i = 0
        for epoch in range(epochs):
            total_rec_loss = 0
            total_rec_doc_loss = 0
            total_clus_loss = 0
            for batch_idx, batch in enumerate(tqdm(dataset_loader, desc="Clustering")):
                optimizer.zero_grad()
                i += 1
                doc_ids = batch[0].to(self.device)
                input_ids = batch[1].to(self.device)
                attention_mask = batch[2].to(self.device)
                valid_pos = batch[3].to(self.device)
                doc_emb, rec_emb, _, p, input_embs, output_embs, rec_doc_emb, p_word = model(input_ids, attention_mask, valid_pos)
                rec_loss = F.mse_loss(output_embs, input_embs)
                rec_doc_loss = F.mse_loss(rec_doc_emb, doc_emb)
                targets = self.target_distribution(p_word).detach()
                clus_loss = F.kl_div(p_word.log(), targets, reduction='batchmean')
                loss = self.args.gamma * clus_loss + rec_loss + self.args.beta * rec_doc_loss
                total_rec_loss += rec_loss.item()
                total_clus_loss += clus_loss.item()
                total_rec_doc_loss += rec_doc_loss.item()
                loss.backward()
                optimizer.step()
            preds = self.inference(topk=20, suffix=f"_{i/len(dataset_loader)}")
            print(f"epoch {epoch}: rec_loss = {total_rec_loss / (batch_idx+1):.4f}; rec_doc_loss = {total_rec_doc_loss / (batch_idx+1):.4f}; cluster_loss = {total_clus_loss / (batch_idx+1):.4f}")

        model_path = os.path.join(self.data_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"model saved to {model_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='yelp')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--n_clusters', default=100, type=int)
    parser.add_argument('--input_dim', default=768, type=int)
    parser.add_argument('--pretrain_epoch', default=100, type=int)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--hidden_dims', default='[500, 500, 1000, 100]', type=str)
    parser.add_argument('--gamma', default=1, type=float, help='weight of clustering loss')
    parser.add_argument('--beta', default=1, type=float, help='weight of clustering loss')
    parser.add_argument('--update_interval', default=500, type=int)
    parser.add_argument('--epochs', default=20, type=int)

    args = parser.parse_args()
    print(args)

    trainer = TopClusTrainer(args)
    pretrained_path = os.path.join("datasets", args.dataset, "pretrained.pt")
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        trainer.model.ae.load_state_dict(torch.load(pretrained_path))
    else:
        print(f"Pretraining autoencoder")
        trainer.pretrain(pretrain_epoch=20)
    trainer.clustering(epochs=args.epochs)
