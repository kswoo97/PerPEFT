### This version uses pre-defined embeddings

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # set this first


import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

import pickle
import time
import math
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, CLIPProcessor, CLIPModel

from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

    
class ClusterwiseProjector(nn.Module) :  ## Below process
    
    def __init__(self, device, dim) :
        super().__init__()
        
        self.proj1 = nn.Linear(dim, dim, bias = False)
        self.ln1 = nn.LayerNorm(dim)
        
    def forward(self, x) : 
        
        newx = self.proj1(x)
        newx = self.ln1(newx)
        
        return newx

class ILSAN (nn.Module) :
    
    def __init__(self, device):
        super().__init__()
    
        self.ImgBlocks = nn.ModuleList()
        self.TextBlocks = nn.ModuleList()
        self.InterBlocks = nn.ModuleList()
        self.CombProj = nn.ModuleList()
        self.n_layers = 7
        self.hid_dim = 32

        for i in range(self.n_layers) : 

            img_block = nn.Sequential(
                nn.Linear(768, self.hid_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.hid_dim, 768, bias=True)
            )
            nn.init.normal_(img_block[0].weight, mean=0.0, std=1e-4)
            nn.init.normal_(img_block[0].bias, mean=0.0, std=1e-4)
            nn.init.normal_(img_block[2].weight, mean=0.0, std=1e-4)
            nn.init.normal_(img_block[2].bias, mean=0.0, std=1e-4)
            self.ImgBlocks.append(img_block)

            # Text Block
            text_block = nn.Sequential(
                nn.Linear(512, self.hid_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.hid_dim, 512, bias=True)
            )
            nn.init.normal_(text_block[0].weight, mean=0.0, std=1e-4)
            nn.init.normal_(text_block[0].bias, mean=0.0, std=1e-4)
            nn.init.normal_(text_block[2].weight, mean=0.0, std=1e-4)
            nn.init.normal_(text_block[2].bias, mean=0.0, std=1e-4)
            self.TextBlocks.append(text_block)

            # Inter Block
            inter_block = nn.Sequential(
                nn.Linear(512, self.hid_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.hid_dim, 512, bias=True)
            )
            nn.init.normal_(inter_block[0].weight, mean=0.0, std=1e-4)
            nn.init.normal_(inter_block[0].bias, mean=0.0, std=1e-4)
            nn.init.normal_(inter_block[2].weight, mean=0.0, std=1e-4)
            nn.init.normal_(inter_block[2].bias, mean=0.0, std=1e-4)
            self.InterBlocks.append(inter_block)
            
            comb_proj = nn.Sequential(
                nn.Linear(768, 512, bias=True)
            )
            self.CombProj.append(comb_proj)
            
        self.weight_scalar = torch.nn.Parameter(torch.zeros((3, self.n_layers)).to(device))
        
    def forward(self, imgX, textX) : # B x 7 x d
        
        cur_weight = torch.sigmoid(self.weight_scalar)
        
        for i in range(self.n_layers) : 
            
            if i == 0 :     
                z_text = textX[:, i, :]
                z_img = imgX[:, i, :]
                z_comb = cur_weight[2, i] * textX[:, i, :] + (1-cur_weight[2, i]) * self.CombProj[i](imgX[:, i, :])
                
            else : 
                z_text = cur_weight[0, i] * z_text + (1-cur_weight[0, i]) * textX[:, i, :]
                z_img = cur_weight[1, i] * z_img + (1-cur_weight[1, i]) * imgX[:, i, :]
                z_comb = cur_weight[2, i] * textX[:, i, :] + (1-cur_weight[2, i]) * self.CombProj[i](imgX[:, i, :]) + z_comb
                
            z_text = self.TextBlocks[i](z_text) + z_text
            z_img = self.ImgBlocks[i](z_img) + z_img
            z_comb = self.InterBlocks[i](z_comb) + z_comb
            
        return torch.hstack([z_text, z_img, z_comb])


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.mmX = None

        self.lin_weight = 0.5

        self.weight_scalar = args.weight_scalar
        
        self.item_emb = torch.nn.Embedding(item_num+1, args.hidden_units, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs, mmX, mm_log_seqs, in_train = True): # TODO: fp64 and int64 as default in python, trim?
            
        seqs = mmX[torch.LongTensor(mm_log_seqs).to(self.dev)]
        seqs = self.weight_scalar * seqs + self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats, mmX

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, 
                mmX, mm_log_seqs, mm_pos_seqs, mm_neg_seqs, 
               give_embs = False): # for training        
        
        log_feats, nX = self.log2feats(log_seqs, mmX, mm_log_seqs) # user_ids hasn't been used yet
        
        pos_embs = self.weight_scalar * nX[torch.LongTensor(mm_pos_seqs).to(self.dev)] + self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.weight_scalar * nX[torch.LongTensor(mm_neg_seqs).to(self.dev)] + self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        if give_embs : 
            
            return log_feats[:, -1, :], pos_logits, neg_logits # pos_pred, neg_pred

        else : 
            
            return pos_logits, neg_logits # pos_pred, neg_pred
            
    def predict(self, user_ids, log_seqs, item_indices): # for inference

        if self.mmX == "None" : 
            print("Assign multimodal feature in self.mmX.")
            pass

        log_feats, _ = self.log2feats(log_seqs, self.mmX, log_seqs, in_train = False) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.weight_scalar * (self.mmX[torch.LongTensor(item_indices).to(self.dev)]) + self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (U, I)
    
    def generate_embeddings(self, user_ids, log_seqs): # for inference

        if self.mmX == "None" : 
            print("Assign multimodal feature in self.mmX.")
            pass

        log_feats, _ = self.log2feats(log_seqs, self.mmX, log_seqs, in_train = False) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        return final_feat
    
class MMProjector(torch.nn.Module):
    
    def __init__(self, args) :
        
        super(MMProjector, self).__init__()

        self.projector = torch.nn.Sequential(torch.nn.Linear(args.in_dim, int(2 * args.hidden_units)), 
                                            torch.nn.LayerNorm(int(2 * args.hidden_units), eps=1e-8), 
                                            torch.nn.ReLU(), 
                                            torch.nn.Linear(int(2 * args.hidden_units), args.hidden_units))
        
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        
    def forward(self, x) : 
        
        return self.last_layernorm(self.projector(x))


def build_index(dataset_name):

    ui_mat = np.loadtxt('./dataset/%s_final.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))

def sample_function2(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, neg_lists):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0 : 
                # neg[idx] = random_neq(1, itemnum + 1, ts)
                neg_idxs = random_neq(0, neg_lists.shape[0], ts)
                neg[idx] = neg_lists[neg_idxs]
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))

        

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
            
class WarpSampler_clusterwise_negative(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, neg_lists = None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function2, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      neg_lists
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1

    f = open('./dataset/%s_final.txt' % fname, 'r')
        
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def seq2idx (seq, pos, neg, idx_mapper) :

    total_all = np.unique(np.hstack([seq, pos, neg]))
    tmp2real_id = []
    tmp2real_dict = {}

    if np.isin(0, total_all) == True : # There is 0

        for iv, rv in enumerate(total_all) : 
            tmp2real_dict[rv] = iv

        
        for item_idx in total_all : 
            
            if item_idx == 0 : 
                tmp2real_id.append(0)
            else : 
                tmp2real_id.append(idx_mapper[item_idx])

    else : # There is no 0

        for iv, rv in enumerate(total_all) : 
            tmp2real_dict[rv] = iv + 1

        tmp2real_id.append(0)
        for item_idx in total_all : 
            tmp2real_id.append(idx_mapper[item_idx])

    new_seq = copy.deepcopy(seq)
    new_pos = copy.deepcopy(pos)
    new_neg = copy.deepcopy(neg)

    for i1 in range(new_seq.shape[0]) : 

        for i2 in range(new_seq.shape[1]) : 

            new_seq[i1, i2] = tmp2real_dict[seq[i1,i2]]
            new_pos[i1, i2] = tmp2real_dict[pos[i1,i2]]
            new_neg[i1, i2] = tmp2real_dict[neg[i1,i2]]
    
    return tmp2real_id, tmp2real_dict, new_seq, new_pos, new_neg

def sub_batch_generator(total_item_lists, sub_batch_size) : 

    if total_item_lists[0] == 0 : ## padding ;
        new_item_sets = total_item_lists[1:]
    else : 
        new_item_sets = total_item_lists
    
    n_splits = math.ceil(len(new_item_sets) / sub_batch_size)
    splits = np.array_split(new_item_sets, n_splits)

    return splits

def itemid2imageandtext(item_list, image_dir, text_info, processor, device):
    
    # Load and preprocess images in batch

    image_inputs = []
    for item in item_list :
        image_path = os.path.join(image_dir, "{0}.jpg".format(item))
        image = Image.open(image_path)
        image_inputs.append(image)

    # Load and tokenize all text descriptions in batch
    text_inputs = [text_info.get(item, "No description available.") for item in item_list]
    
    inputs = processor(text=text_inputs,
                       images=image_inputs,
                       return_tensors="pt",padding=True, 
                      truncation=True,        # ensures text is cut off if too long
                        max_length=77)           # 77 tokens is standard for CLIP)
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
    
    return inputs

def multimodal_encoding(encoder, inputs, is_trained = False) : 
    
    if is_trained : 
        outputs = encoder(**inputs)

    else : 
        with torch.no_grad() : 
            encoder.eval()
            outputs = encoder(**inputs)
            
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    return torch.hstack([image_embeds, text_embeds])

def get_multimodal_features (total_item_lists, sub_batch_size, 
                             image_dir, text_info, processor, device, 
                             encoder, in_dim = 1024, is_trained = True) : 
    
    ## 1024 (SwinT) + 768 (RoBERTa) = 1792
    
    sub_batches = sub_batch_generator(total_item_lists, sub_batch_size)
    TX = torch.zeros((len(total_item_lists), in_dim), dtype = torch.float32).to(device)
    prev_idx = 0

    if total_item_lists[0] == 0 :  # There is a padding
        strider = 1
    else : 
        strider = 0

    for idx, tmp_batch in (enumerate(sub_batches)) : 

        inputs = itemid2imageandtext(tmp_batch, image_dir, 
                                              text_info, processor, device)
        curX = multimodal_encoding(encoder, inputs, is_trained)
        TX[prev_idx + strider : prev_idx + curX.shape[0] + strider] = curX
        prev_idx += curX.shape[0]

        del curX

    return TX

# evaluate on val set
def evaluate_mm_valid(model, projector, dataset, args, total_item_lists,
                image_dir, text_info, processor, device,
                encoder, in_dim = 1024, give_user_n = False, abl = "no"):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    NDCG1 = 0.0
    HT1 = 0.0

    NDCG5 = 0.0
    HT5 = 0.0

    NDCG10 = 0.0
    HT10 = 0.0

    random.seed(0)
    
    users = range(1, usernum + 1)

    # item_idx = list(np.arange(1, itemnum + 1))
    total_items = set(range(1, itemnum + 1))

    sub_batches = sub_batch_generator(total_item_lists, 512)
    TX = torch.zeros((len(total_item_lists), in_dim), dtype = torch.float32).to(device)
    prev_idx = 0

    if total_item_lists[0] == 0 :  # There is a padding
        strider = 1
    else : 
        strider = 0

    with torch.no_grad() : 

        encoder.eval()
        model.eval()
        projector.eval()

        print(len(sub_batches))

        for idx, tmp_batch in tqdm(enumerate(sub_batches)) : 

            inputs = itemid2imageandtext(tmp_batch, image_dir, text_info, processor, device)
            curX = multimodal_encoding(encoder, inputs, False) # No training
            
            TX[prev_idx + strider : prev_idx + curX.shape[0] + strider] = curX
            prev_idx += curX.shape[0]

            del curX 
            
        if abl == "text" : 
            TX = TX[:, 512:]
            
        elif abl == "image" : 
            TX = TX[:, :512]
            
        TX = projector(TX)

        model.mmX = TX # Assign multi-modal features!
    
        for u in users:
            
            if len(train[u]) < 1 or len(valid[u]) < 1: continue
    
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
    
            rated = set(train[u])
            rated.add(0)
    
            item_idx = [test[u][0]]
            item_idx = item_idx + list(total_items - {test[u][0]} - rated)
    
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0]
    
            rank = predictions.argsort().argsort()[0].item()
    
            valid_user += 1
    
            if rank < 10:
                NDCG1 += 1 / np.log2(rank + 2)
                HT1 += 1
    
            if rank < 20:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1
    
            if rank < 30:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1
    
            
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    if give_user_n :
        return valid_user, [NDCG1, NDCG5, NDCG10, HT1, HT5, HT10], TX
    else: 
        return [(NDCG1 / valid_user), (NDCG5 / valid_user), (NDCG10 / valid_user), (HT1 / valid_user), (HT5 / valid_user), (HT10 / valid_user)], TX

def evaluate_mm_test(model, projector, dataset, args, total_item_lists,
                image_dir, text_info, processor, device, 
                encoder, in_dim = 1024, give_user_n = False, inX = None, print_time= False):
    
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    NDCG1 = 0.0
    HT1 = 0.0

    NDCG5 = 0.0
    HT5 = 0.0

    NDCG10 = 0.0
    HT10 = 0.0

    random.seed(0)
    
    users = range(1, usernum + 1)
    
    
    # item_idx = list(np.arange(1, itemnum + 1))
    total_items = set(range(1, itemnum + 1))

    sub_batches = sub_batch_generator(total_item_lists, 512)
    TX = torch.zeros((len(total_item_lists), in_dim), dtype = torch.float32).to(device)
    prev_idx = 0

    if total_item_lists[0] == 0 :  # There is a padding
        strider = 1
    else : 
        strider = 0

    with torch.no_grad() : 

        encoder.eval()
        model.eval()
        projector.eval()
        
        if inX is not None : 
            
            TX = inX
            
        else : 

            print(len(sub_batches))

            for idx, tmp_batch in tqdm(enumerate(sub_batches)) : 

                inputs = itemid2imageandtext(tmp_batch, image_dir, text_info, processor, device)
                curX = multimodal_encoding(encoder, inputs, False) # No training
                TX[prev_idx + strider : prev_idx + curX.shape[0] + strider] = curX
                prev_idx += curX.shape[0]

                del  curX

            TX = projector(TX)

        model.mmX = TX # Assign multi-modal features!
        
        TT11 = time.time()
    
        for u in users :
            
            if len(train[u]) < 1 or len(valid[u]) < 1: continue
    
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            
            for i in reversed(train[u]) :
                
                seq[idx] = i
                idx -= 1
                if idx == -1: break
                    
            rated = set(train[u]).union({valid[u][0]})
            rated.add(0)
            item_idx = [test[u][0]]
            item_idx = item_idx + list(total_items - {test[u][0]} - rated)
    
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0]
    
            rank = predictions.argsort().argsort()[0].item()
    
            valid_user += 1
    
            if rank < 10:
                NDCG1 += 1 / np.log2(rank + 2)
                HT1 += 1
    
            if rank < 20:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1
    
            if rank < 30:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1
    
            
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    # return [(NDCG1 / valid_user), (NDCG5 / valid_user), (NDCG10 / valid_user), (HT1 / valid_user), (HT5 / valid_user), (HT10 / valid_user)]
    TT22 = time.time()
    
    if print_time :
        print("Time")
        print(TT22 - TT11)
    if give_user_n :
        return valid_user, [NDCG1, NDCG5, NDCG10, HT1, HT5, HT10]
    else: 
        return [(NDCG1 / valid_user), (NDCG5 / valid_user), (NDCG10 / valid_user), (HT1 / valid_user), (HT5 / valid_user), (HT10 / valid_user)]
    

# evaluate on val set
def evaluate_mm_ilsan_valid(model, projector, dataset, args, total_item_lists, id2ilsanidx,
                imgX, textX, device, encoder, in_dim = 1024, give_user_n = False) :
    
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    NDCG1 = 0.0
    HT1 = 0.0

    NDCG5 = 0.0
    HT5 = 0.0

    NDCG10 = 0.0
    HT10 = 0.0

    random.seed(0)
    
    users = range(1, usernum + 1)

    # item_idx = list(np.arange(1, itemnum + 1))
    total_items = set(range(1, itemnum + 1))

    sub_batches = sub_batch_generator(total_item_lists, 512)
    TX = torch.zeros((len(total_item_lists), in_dim), dtype = torch.float32).to(device)
    prev_idx = 0

    if total_item_lists[0] == 0 :  # There is a padding
        strider = 1
    else : 
        strider = 0

    with torch.no_grad() : 

        encoder.eval()
        model.eval()

        print(len(sub_batches))

        for idx, tmp_batch in tqdm(enumerate(sub_batches)) : 

            new_mapper = [id2ilsanidx[ii] for ii in tmp_batch]
            tmpimgX, tmptextX = imgX[new_mapper, :, :], textX[new_mapper, :, :]
            curX = encoder(tmpimgX, tmptextX)
            
            TX[prev_idx + strider : prev_idx + curX.shape[0] + strider] = curX
            prev_idx += curX.shape[0]

            del curX 

        TX = projector(TX)
        
        # TX = model.init_layernorm(model.projector(TX))

        model.mmX = TX # Assign multi-modal features!
    
        for u in users:
            
            if len(train[u]) < 1 or len(valid[u]) < 1: continue
    
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
    
            rated = set(train[u])
            rated.add(0)
    
            item_idx = [test[u][0]]
            item_idx = item_idx + list(total_items - {test[u][0]} - rated)
    
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0]
    
            rank = predictions.argsort().argsort()[0].item()
    
            valid_user += 1
    
            if rank < 10:
                NDCG1 += 1 / np.log2(rank + 2)
                HT1 += 1
    
            if rank < 20:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1
    
            if rank < 30:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1
    
            
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    if give_user_n :
        return valid_user, [NDCG1, NDCG5, NDCG10, HT1, HT5, HT10], TX
    else: 
        return [(NDCG1 / valid_user), (NDCG5 / valid_user), (NDCG10 / valid_user), (HT1 / valid_user), (HT5 / valid_user), (HT10 / valid_user)], TX

def evaluate_mm_ilsan_test(model, projector, dataset, args, total_item_lists, id2ilsanidx,
                imgX, textX, device, encoder, in_dim = 1024, give_user_n = False, inX = None):
    
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0

    NDCG1 = 0.0
    HT1 = 0.0

    NDCG5 = 0.0
    HT5 = 0.0

    NDCG10 = 0.0
    HT10 = 0.0

    random.seed(0)
    
    users = range(1, usernum + 1)

    # item_idx = list(np.arange(1, itemnum + 1))
    total_items = set(range(1, itemnum + 1))

    sub_batches = sub_batch_generator(total_item_lists, 512)
    TX = torch.zeros((len(total_item_lists), in_dim), dtype = torch.float32).to(device)
    prev_idx = 0

    if total_item_lists[0] == 0 :  # There is a padding
        strider = 1
    else : 
        strider = 0

    with torch.no_grad() : 

        encoder.eval()
        model.eval()
        
        if inX is not None : 
            
            TX = inX
            
        else : 

            print(len(sub_batches))

            for idx, tmp_batch in tqdm(enumerate(sub_batches)) : 

                new_mapper = [id2ilsanidx[ii] for ii in tmp_batch]
                tmpimgX, tmptextX = imgX[new_mapper, :, :], textX[new_mapper, :, :]
                curX = encoder(tmpimgX, tmptextX)
                TX[prev_idx + strider : prev_idx + curX.shape[0] + strider] = curX
                prev_idx += curX.shape[0]

                del  curX

            TX = projector(TX)
            
            # TX = model.init_layernorm(model.projector(TX))

        model.mmX = TX # Assign multi-modal features!
    
        for u in users :
            
            if len(train[u]) < 1 or len(valid[u]) < 1: continue
    
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            
            for i in reversed(train[u]) :
                
                seq[idx] = i
                idx -= 1
                if idx == -1: break
                    
            rated = set(train[u]).union({valid[u][0]})
            rated.add(0)
            item_idx = [test[u][0]]
            item_idx = item_idx + list(total_items - {test[u][0]} - rated)
    
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0]
    
            rank = predictions.argsort().argsort()[0].item()
    
            valid_user += 1
    
            if rank < 10:
                NDCG1 += 1 / np.log2(rank + 2)
                HT1 += 1
    
            if rank < 20:
                NDCG5 += 1 / np.log2(rank + 2)
                HT5 += 1
    
            if rank < 30:
                NDCG10 += 1 / np.log2(rank + 2)
                HT10 += 1
    
            
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    # return [(NDCG1 / valid_user), (NDCG5 / valid_user), (NDCG10 / valid_user), (HT1 / valid_user), (HT5 / valid_user), (HT10 / valid_user)]
    if give_user_n :
        return valid_user, [NDCG1, NDCG5, NDCG10, HT1, HT5, HT10]
    else: 
        return [(NDCG1 / valid_user), (NDCG5 / valid_user), (NDCG10 / valid_user), (HT1 / valid_user), (HT5 / valid_user), (HT10 / valid_user)]
    
def get_user_embeddings(model, dataset, args) :

    model.eval()

    with torch.no_grad() : 
    
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

        TX = torch.zeros((usernum + 1, args.hidden_units))
        
        valid_user = 0.0
    
        all_users = np.arange(1, usernum+1)
    
        total_items = set(range(1, itemnum + 1))
            
        for u in tqdm(all_users):
    
            if len(train[u]) < 1 or len(test[u]) < 1: continue
    
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]
    
            x = model.generate_embeddings(*[np.array(l) for l in [[u], [seq]]])

            TX[u, :] = x

    return TX