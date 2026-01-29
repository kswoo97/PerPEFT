import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # set this first

import time
import torch
import argparse
import copy

from tqdm import tqdm


from torch.cuda.amp import autocast, GradScaler

from peft import PeftModel
import hashlib

from typing import Dict, Any

from sklearn.cluster import KMeans

from perpeft_utils import *
from perpeft_gradient_changer import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', '--dataset', default="toys_games", type = str)
parser.add_argument('-peft', '--peft_type', default="iisan", type=str)
parser.add_argument('-device', '--device', default='cuda:0', type=str)
parser.add_argument('-trial', '--trial', default=0, type=int)
parser.add_argument('--train_dir', default="./dataset", type = str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--maxlen', default=10, type=int)
parser.add_argument('--hidden_units', default=32, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--num_heads', default=4, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('--l2_emb', default=1e-6, type=float)
parser.add_argument('--weight_scalar', default=0.5, type=float)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--mid_reduce', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--starting_epoch', default=10, type=int)
parser.add_argument('--wdecay', default=1e-4, type=float)
parser.add_argument('--rank', default=4, type=int)
parser.add_argument('--C', default=8, type=int)

args = parser.parse_args()

args.in_dim = 1792 ## Dimension of IISAN's input feature.

if __name__ == '__main__':
    
    os.makedirs("pretrained_models", exist_ok=True)
    
    inter_epoch = 10
    
    ##### Step 1: GLOBAL PEFT
    
    u2i_index, i2u_index = build_index(args.dataset)

    scaler = GradScaler()  # Handles dynamic loss scaling
    
    # global dataset
    dataset = data_partition(args.dataset)

    with open("./dataset/{0}_item2id_final.pickle".format(args.dataset), "rb") as f :
        idx2item = pickle.load(f)

    with open("./dataset/{0}_texts_final.pickle".format(args.dataset), "rb") as f :
        text_info = pickle.load(f)
        
    TotalImageX, TotalTextX, IlsanIDX = torch.load("./dataset/{0}_preX.pt".format(args.dataset))
    
    IlsanMapper = {idx : v for v, idx in enumerate(IlsanIDX)}
    
    TotalImageX, TotalTextX = TotalImageX.to(args.device), TotalTextX.to(args.device)

    device = args.device

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    
    ## Load models
    torch.manual_seed(0)
    
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    projector = MMProjector(args).to(args.device)
    
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    encoder = IISAN(args.device).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    model.train() # enable model training

    epoch_start_idx = 1

    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()

    adam_optimizer = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=args.lr, 
                                       weight_decay = args.wdecay)

    backbone_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, 
                                           weight_decay = args.wdecay)

    total_item_lists = [0]

    for k in idx2item : 

        total_item_lists.append(idx2item[k])


    for epoch in range(1, 11) : # 10 pre-training epochs ## 11

        pos_labels, neg_labels = torch.ones(5, device=args.device), torch.zeros(5, device=args.device)

        if args.inference_only: break # just to decrease identition

        total_losses = 0.0

        torch.manual_seed(epoch)

        for step in tqdm(range(num_batch)) : # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):

            model.train()
            encoder.train()
            projector.train()

            adam_optimizer.zero_grad()
            backbone_optimizer.zero_grad()

            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            tmp2real_id, tmp2real_dict, new_seq, new_pos, new_neg = seq2idx (seq = seq, pos = pos, neg = neg, idx_mapper = idx2item)

            with autocast():

                current_idxs = [IlsanMapper[tidid] for tidid in tmp2real_id]
                curX = encoder(TotalImageX[current_idxs], TotalTextX[current_idxs])
                curX = projector(curX)

                pos_logits, neg_logits = model(u, seq, pos, neg, curX, new_seq, new_pos, new_neg)

                if (pos_labels.shape == pos_logits) & (neg_labels.shape == neg_logits) : # No need to define a new label
                    None

                else: # Update!
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

                indices = np.where(pos != 0)

                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters() : 
                    loss += args.l2_emb * torch.norm(param)

            scaler.scale(loss).backward()
            scaler.step(adam_optimizer)
            scaler.step(backbone_optimizer)
            scaler.update()

            total_losses += loss.detach().cpu().item()

            del curX, pos_logits, neg_logits
    

        torch.save(model.state_dict(), f"./pretrained_models/sasrec_{args.peft_type}_{args.dataset}_{epoch}_{args.lr}_{args.wdecay}.pth")
        torch.save(projector.state_dict(), f"./pretrained_models/projector_{args.peft_type}_{args.dataset}_{epoch}_{args.lr}_{args.wdecay}.pth")
        torch.save(encoder.state_dict(), f"./pretrained_models/encoder_{args.peft_type}_{args.dataset}_{epoch}_{args.lr}_{args.wdecay}.pth")
    
    #### STEP 2: User Grouping 
    
    model.eval()
    projector.eval()
    encoder.eval()

    total_items = set(range(1, itemnum + 1))

    in_dim = 1792

    sub_batches = sub_batch_generator(total_item_lists, 2048)
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

        for idx, tmp_batch in (enumerate(sub_batches)) : 

            new_mapper = [IlsanMapper[ii] for ii in tmp_batch]
            tmpimgX, tmptextX = TotalImageX[new_mapper, :, :], TotalTextX[new_mapper, :, :]
            curX = encoder(tmpimgX, tmptextX)
            TX[prev_idx + strider : prev_idx + curX.shape[0] + strider] = curX
            prev_idx += curX.shape[0]

            del curX 

        TX = projector(TX)

    model.mmX = TX # Assign multi-modal features!

    cx = get_user_embeddings(model, dataset, args)
    newX = cx.cpu().numpy()

    nC = 8

    kmeans = KMeans(n_clusters = nC, max_iter = 100).fit(newX)
    labels = kmeans.labels_

    user_type_dict = {i: v for i, v, in enumerate(labels)}
    unique_user_ids = [dict() for _ in range(nC)]
    idxs = [1 for _ in range(nC)]

    for i in user_type_dict : 

        curC = user_type_dict[i]

        if i == 0 : 
            continue

        unique_user_ids[curC][i] = idxs[curC]

        idxs[curC] += 1

    input_file = "./dataset/{0}_final.txt".format(args.dataset)

    C = max(user_type_dict.values()) + 1
    output_files = [open(f"./dataset/{args.dataset}_{args.peft_type}_{t}_final.txt", "w") for t in range(C)]

    with open(input_file, "r") as f:

        for i in range(labels.shape[0]) : 

            if i == 0 : 
                continue
            else : 
                total_lines = dataset[0][i] + dataset[1][i] + dataset[2][i]

            user_type = user_type_dict[i]

            for item_id_str in total_lines : 

                user_id = int(i)

                if user_id == 0 : 
                    continue

                new_line = "{0} {1}\n".format(unique_user_ids[user_type][user_id], item_id_str)

                if user_id in user_type_dict:
                    user_type = user_type_dict[user_id]
                    output_files[user_type].write(new_line)
                else:
                    print(f"User {user_id} not in user_type_dict, skipping.")

    for f in output_files:
        f.close()
    
    ##### Step 3: Personalized PEFT
    
    Datasets = []
    Samplers = []
    
    device = args.device
    
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    with open("./dataset/{0}_item2id_final.pickle".format(args.dataset), "rb") as f :
        idx2item = pickle.load(f)

    with open("./dataset/{0}_texts_final.pickle".format(args.dataset), "rb") as f :
        text_info = pickle.load(f)
        
    scaler = GradScaler()  # Handles dynamic loss scaling
    
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    encoder = CLIPModel.from_pretrained(model_name).to(args.device)
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    model_param = torch.load(f"./pretrained_models/sasrec_{args.peft_type}_{args.dataset}_{inter_epoch}_{args.lr}_{args.wdecay}.pth")
    model.load_state_dict(copy.deepcopy(model_param))
    
    all_projectors = []
    all_encoders = []
    
    n_batches = {i:0 for i in range(args.C)}
    
    for t in range(args.C) :  ## Do this iteratively across clusters

        tmp_dataset = data_partition(args.dataset + "_{1}_{0}".format(t, args.peft_type)) # Load dataset accordingly to the clusters
        
        [c_user_train, c_user_valid, c_user_test, c_usernum, c_itemnum] = tmp_dataset
        
        neg_lists = []
        for user_neg_id in c_user_train : 
            neg_lists.extend(c_user_train[user_neg_id])
        neg_lists = np.array(list(set(neg_lists)))
        
        tmp_sampler = WarpSampler_clusterwise_negative(c_user_train, c_usernum, c_itemnum, batch_size=args.batch_size, maxlen=args.maxlen, 
                                                       n_workers=3, neg_lists = neg_lists)
        
        c_batch = (len(c_user_train) - 1) // args.batch_size + 1
        n_batches[t] = c_batch
        print(c_batch)
        
        Datasets.append(tmp_dataset)
        Samplers.append(tmp_sampler)
        
        cur_projector = MMProjector(args).to(args.device)
        cur_proj_param = torch.load(f"./pretrained_models/projector_{args.peft_type}_{args.dataset}_{inter_epoch}_{args.lr}_{args.wdecay}.pth")
        cur_projector.load_state_dict(copy.deepcopy(cur_proj_param))
        all_projectors.append(cur_projector)
        
        cur_encoder = IISAN(args.device).to(args.device)
        encoder_param = torch.load(f"./pretrained_models/encoder_{args.peft_type}_{args.dataset}_{inter_epoch}_{args.lr}_{args.wdecay}.pth")
        cur_encoder.load_state_dict(copy.deepcopy(encoder_param))
        all_encoders.append(cur_encoder)
    
    epoch_start_idx = 1
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    
    all_params = []
    all_params += list(model.parameters())
    
    for cur_proj in all_projectors :     
        all_params += list(cur_proj.parameters())
    
    all_enc_params = []
    for c_enc in all_encoders : 
        all_enc_params += list(c_enc.parameters())
    
    adam_optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay = args.wdecay)
    backbone_optimizer = torch.optim.AdamW(all_enc_params, lr=args.lr, weight_decay = args.wdecay)

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    
    total_all_results = dict()

    total_item_lists = [0]

    for k in idx2item : 
    
        total_item_lists.append(idx2item[k])
    
    batch_indexer = []
    
    for t in range(args.C) : 
    
        batch_indexer.extend([t] * n_batches[t])
        
    batch_indexer = np.array(batch_indexer)
    print(batch_indexer.shape)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        
        T1 = time.time()
        
        pos_labels, neg_labels = torch.ones(5, device=args.device), torch.zeros(5, device=args.device)
        
        if args.inference_only: break # just to decrease identition

        total_losses = 0.0
        
        np.random.seed(epoch)
        np.random.shuffle(batch_indexer)
        
        for idid, t in tqdm(enumerate(batch_indexer)) : 

            sampler = Samplers[t]
            encoder = all_encoders[t]
            projector = all_projectors[t]

            model.train()
            encoder.train()
            projector.train()

            adam_optimizer.zero_grad()
            backbone_optimizer.zero_grad()

            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            tmp2real_id, tmp2real_dict, new_seq, new_pos, new_neg = seq2idx (seq = seq, pos = pos, neg = neg, idx_mapper = idx2item)

            with autocast():

                current_idxs = [IlsanMapper[tidid] for tidid in tmp2real_id]
                curX = encoder(TotalImageX[current_idxs], TotalTextX[current_idxs])
                curX = projector(curX)

                embs, pos_logits, neg_logits = model(u, seq, pos, neg, curX, new_seq, new_pos, new_neg, give_embs = True)
                
                if (pos_labels.shape == pos_logits) & (neg_labels.shape == neg_logits) : # No need to define a new label
                    None
                
                else: # Update!
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

                indices = np.where(pos != 0)

                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                
            for param in model.item_emb.parameters() : 
                loss += args.l2_emb * torch.norm(param)

            # loss.backward()
            # adam_optimizer.step()
            # backbone_optimizer.step()

            scaler.scale(loss).backward()
            safe_step(scaler, adam_optimizer)
            safe_step(scaler, backbone_optimizer)
            scaler.update()

            total_losses += loss.detach().cpu().item()

            del curX, pos_logits, neg_logits
        
        V1 = 0.0
        V2 = 0.0
        V3 = 0.0
        V4 = 0.0
        V5 = 0.0
        V6 = 0.0
        
        T1 = 0.0
        T2 = 0.0
        T3 = 0.0
        T4 = 0.0
        T5 = 0.0
        T6 = 0.0
        
        N1 = 0.0 ; N2 = 0.0
        
        model.eval()
        
        for t in range(args.C) : 
            
            dataset = Datasets[t]
            
            encoder = all_encoders[t]
            encoder.eval()
            
            projector = all_projectors[t]
            projector.eval()
            
            [c_user_train, c_user_valid, c_user_test, c_usernum, c_itemnum] = dataset
            
            n_users_valid, t_valid, mmX = evaluate_mm_ilsan_valid(model, projector, dataset, args, total_item_lists,
                             IlsanMapper, TotalImageX, TotalTextX, device = device, 
                             encoder = encoder, in_dim = 1792, give_user_n = True)

            n_users_test, t_test = evaluate_mm_ilsan_test(model, projector, dataset, args, total_item_lists,
                             IlsanMapper, TotalImageX, TotalTextX, device = device, 
                             encoder = encoder, in_dim = 1792, inX = mmX, give_user_n = True)

            # N1 += n_users_valid
            N1 += n_users_valid
            N2 += n_users_test

            V1 += t_valid[0] ; V2 += t_valid[1] ; V3 += t_valid[2] ; V4 += t_valid[3] ; V5 += t_valid[4] ; V6 += t_valid[5] ;
            T1 += t_test[0] ; T2 += t_test[1] ; T3 += t_test[2] ; T4 += t_test[3] ; T5 += t_test[4] ; T6 += t_test[5] ;

        print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, NDCG@20: %.4f, NDCG@30: %.4f, HR@10: %.4f, HR@20: %.4f, HR@30: %.4f), test (NDCG@10: %.4f, NDCG@20: %.4f, NDCG@30: %.4f, HR@10: %.4f, HR@20: %.4f, HR@30: %.4f)'% (epoch, T, (V1/N1), (V2/N1), (V3/N1), (V4/N1), (V5/N1), (V6/N1), (T1/N2), (T2/N2), (T3/N2), (T4/N2), (T5/N2), (T6/N2)))

    sampler.close()
    print("Done")
