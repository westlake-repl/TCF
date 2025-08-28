import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .dataset import SeqEvalDataset, PointWiseEvalDataset, SequentialDistributedSampler
import torch.distributed as dist
import math
from tqdm import tqdm


class ModalItemDataset(Dataset):
    def __init__(self, attention_mask, inputs_hidden_state):
        super().__init__()
        self.attention_mask = attention_mask
        self.inputs_hidden_state = inputs_hidden_state
        
    def __getitem__(self, idx):
        return self.attention_mask[idx], self.inputs_hidden_state[idx]
    
    def __len__(self):
        return self.inputs_hidden_state.shape[0]

class IterDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def collate_fn_(arr):
    arr = torch.LongTensor(arr)
    return arr


def print_metrics(x, Log_file, v_or_t):
    Log_file.info(v_or_t+"_results   {}".format('\t'.join(["{:0.5f}".format(i * 100) for i in x])))


def get_mean(arr):
    return [i.mean() for i in arr]


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


def eval_concat(eval_list, test_sampler):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = distributed_concat(eval_m, len(test_sampler.dataset))\
            .to(torch.device("cpu")).numpy()
        eval_result.append(eval_m_cpu.mean())
    return eval_result


def metrics_topK(y_score, y_true, item_rank, topK, local_rank):
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)
    rank = torch.sum(y_true * item_rank)
    eval_ra = torch.zeros(2).to(local_rank)
    if rank <= topK:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    return eval_ra


def get_item_embeddings(model, item_content, test_batch_size, args, use_modal, local_rank):
    model.eval()
    if use_modal:
        if model.module.text_encoder.text_model is not None:
            attention_mask, inputs_hidden_state = item_content
            item_dataset = ModalItemDataset(attention_mask, inputs_hidden_state)
            item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                        pin_memory=True)
        else:
           item_dataset = IterDataset(item_content)
           item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                        pin_memory=True) 
    else:
        item_dataset = IterDataset(item_content)
        item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                    pin_memory=True, collate_fn=collate_fn_)
        
    item_embeddings = []
    with torch.no_grad():
        for data in item_dataloader:
            if use_modal:
                if model.module.text_encoder.text_model is not None:
                    attention_mask, inputs_hidden_state = data
                    input_attn_mask_hidden_state = attention_mask.to(local_rank), inputs_hidden_state.to(local_rank)
                    item_emb = model.module.text_encoder(input_attn_mask_hidden_state)
                else:
                    item_emb = data.to(local_rank)
                    item_emb = model.module.text_encoder(item_emb)
            else:
                input_ids = data.to(local_rank)
                item_emb = model.module.id_embedding(input_ids)
                    
            if args.architecture == "DSSM":
                item_emb = model.module.item_encoder(item_emb)
            
            item_embeddings.extend(item_emb)
    return torch.stack(tensors=item_embeddings, dim=0).to(torch.device("cpu")).detach()


def get_user_embeddings(model, user_num, test_batch_size, args, local_rank):
    model.eval()
    user_dataset = IterDataset(data=np.arange(user_num + 1))
    user_dataloader = DataLoader(user_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True, collate_fn=collate_fn_)
    user_embeddings = []
    with torch.no_grad():
        for input_ids in user_dataloader:
            input_ids = input_ids.to(local_rank)
            user_emb = model.module.user_embedding(input_ids)
            user_feature = model.module.user_encoder(user_emb)
            user_embeddings.extend(user_feature)
    return torch.stack(tensors=user_embeddings, dim=0).to(torch.device("cpu")).detach()


def eval_sasrec_model(model, user_history, eval_seq, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = SeqEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    
    if v_or_t == 'train' and args.valid_data_num > 0:
            # generate random indices no repeat
            eval_indices = np.random.choice(len(eval_dataset), args.valid_data_num, replace=False)
            eval_dataset = torch.utils.data.Subset(eval_dataset, eval_indices)
    
    if v_or_t == 'test' and args.test_data_num > 0:
            # generate random indices no repeat
            eval_indices = np.random.choice(len(eval_dataset), args.test_data_num, replace=False)
            eval_dataset = torch.utils.data.Subset(eval_dataset, eval_indices)


    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        show_progress = args.show_progress
        dl = tqdm(
            eval_dl,
            total=len(eval_dl),
            ncols=100,
            desc=f"Eval {v_or_t}",
        ) if show_progress.lower() == 'true' else eval_dl
        for data in dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval


def eval_dssm_model(model, user_history, eval_pairs, user_embeddings, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = PointWiseEvalDataset(eval_pairs=eval_pairs, user_content=user_embeddings, item_num=item_num)
    
    if v_or_t == 'train' and args.valid_data_num > 0:
            # generate random indices no repeat
            eval_indices = np.random.choice(len(eval_dataset), args.valid_data_num, replace=False)
            eval_dataset = torch.utils.data.Subset(eval_dataset, eval_indices)
    
    if v_or_t == 'test' and args.test_data_num > 0:
            # generate random indices no repeat
            eval_indices = np.random.choice(len(eval_dataset), args.test_data_num, replace=False)
            eval_dataset = torch.utils.data.Subset(eval_dataset, eval_indices)

    
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + '_methods   {}'.format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        show_progress = args.show_progress
        dl = tqdm(
            eval_dl,
            total=len(eval_dl),
            ncols=100,
            desc=f"Eval {v_or_t}",
        ) if show_progress.lower() == 'true' else eval_dl
        for data in dl:
            user_ids, user_embs, labels = data
            user_ids, user_embs, labels = \
                user_ids.to(local_rank), user_embs.to(local_rank), labels.to(local_rank)
            scores = torch.matmul(user_embs, item_embeddings.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval
