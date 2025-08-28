import torch
from torch.utils.data import Dataset
import numpy as np
import random
import math

class TextSeqNegSampleAllFreezeTrainDataset(Dataset):
    def __init__(self, u2seq, item_embs, item_num, max_seq_len, neg_num=1):
        super(TextSeqNegSampleAllFreezeTrainDataset, self).__init__()
        self.behaviors = list(u2seq.items())
        self.item_embs = item_embs
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        assert neg_num == 1, "only support one negative sample when performing sequential training now"
        self.neg_num = neg_num

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        user_id, seq = self.behaviors[idx] 
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        neg_items = []
        for _ in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        
        pos_item_id = torch.LongTensor([0] * mask_len_head + seq)
        neg_item_id = torch.LongTensor([0] * (mask_len_head + 1) + neg_items)
        pos_item_emb = self.item_embs[pos_item_id]
        neg_item_emb = self.item_embs[neg_item_id]
        mask = torch.FloatTensor(log_mask)
        
        return pos_item_id, neg_item_id, pos_item_emb, neg_item_emb, mask


class TextSeqNegSampleFinetuneTrainDataset(Dataset):
    def __init__(self, u2seq, attention_mask, inputs_hidden_state, item_num, max_seq_len, neg_num=1):
        super(TextSeqNegSampleFinetuneTrainDataset, self).__init__()
        self.behaviors = list(u2seq.items())
        self.attention_mask = attention_mask
        self.inputs_hidden_state = inputs_hidden_state
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        assert neg_num == 1, "only support one negative sample when performing sequential training now"
        self.neg_num = neg_num
        
    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        user_id, seq = self.behaviors[idx] 
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        neg_items = []
        for _ in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        pos_item_id = torch.LongTensor([0] * mask_len_head + seq)
        neg_item_id = torch.LongTensor([0] * (mask_len_head + 1) + neg_items)
        pos_attention_mask = self.attention_mask[pos_item_id]
        neg_attention_mask = self.attention_mask[neg_item_id]
        pos_inputs_hidden_state = self.inputs_hidden_state[pos_item_id]
        neg_inputs_hidden_state = self.inputs_hidden_state[neg_item_id]
        mask = torch.FloatTensor(log_mask)
        return pos_item_id, neg_item_id, pos_attention_mask, neg_attention_mask, pos_inputs_hidden_state, neg_inputs_hidden_state, mask


class IdSeqNegSampleTrainDataset(Dataset):
    def __init__(self, u2seq, item_content, item_num, max_seq_len, neg_num=1):
        super(IdSeqNegSampleTrainDataset, self).__init__()
        self.behaviors = list(u2seq.items())
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        assert neg_num == 1, "only support one negative sample when performing sequential training now"
        self.neg_num = neg_num

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        user_id, seq = self.behaviors[idx] 
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        neg_items = []
        for _ in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        pos_item_id = torch.LongTensor([0] * mask_len_head + seq)
        neg_item_id = torch.LongTensor([0] * (mask_len_head + 1) + neg_items)
        mask = torch.FloatTensor(log_mask)
        return pos_item_id, neg_item_id, mask


class TextPointWiseNegSampleAllFreezeDataset(Dataset):
    def __init__(self, user_history, train_pairs, item_num, item_embs, neg_num=1):
        super(TextPointWiseNegSampleAllFreezeDataset, self).__init__()
        self.user_history = user_history
        self.train_pairs = train_pairs
        self.item_num = item_num
        self.item_embs = item_embs
        self.neg_num = neg_num
        
    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, index):
        (user_id, pos_id) = self.train_pairs[index]
        history = self.user_history[user_id]
        neg_items = []
        for _ in range(self.neg_num):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in history:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        pos_item_id = torch.LongTensor([pos_id])
        neg_item_id = torch.LongTensor(neg_items)
        pos_item_emb = self.item_embs[pos_item_id]
        neg_item_emb = self.item_embs[neg_item_id]
        return user_id, pos_item_emb, neg_item_emb


class TextPointWiseNegSampleFinetuneDataset(Dataset):
    def __init__(self, user_history, train_pairs, item_num, attention_mask, inputs_hidden_state, neg_num=1):
        self.user_history = user_history
        self.train_pairs = train_pairs
        self.item_num = item_num
        self.attention_mask = attention_mask
        self.inputs_hidden_state = inputs_hidden_state
        self.neg_num = neg_num
        
    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, index):
        (user_id, pos_id) = self.train_pairs[index]
        history = self.user_history[user_id]
        neg_items = []
        for _ in range(self.neg_num):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in history:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        pos_item_id = torch.LongTensor([pos_id])
        neg_item_id = torch.LongTensor(neg_items)
        pos_attention_mask = self.attention_mask[pos_item_id]
        neg_attention_mask = self.attention_mask[neg_item_id]
        pos_inputs_hidden_state = self.inputs_hidden_state[pos_item_id]
        neg_inputs_hidden_state = self.inputs_hidden_state[neg_item_id]
        return user_id, pos_attention_mask, pos_inputs_hidden_state, neg_attention_mask, neg_inputs_hidden_state
        

class IdPointWiseNegSampleDataset(Dataset):
    def __init__(self, user_history, train_pairs, item_num, neg_num=1):
        self.user_history = user_history
        self.train_pairs = train_pairs
        self.item_num = item_num
        self.neg_num = neg_num
        
    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, index):
        (user_id, pos_id) = self.train_pairs[index]
        history = self.user_history[user_id]
        neg_items = []
        for _ in range(self.neg_num):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in history:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
       
        pos_item_id = torch.LongTensor([pos_id])
        neg_item_id = torch.LongTensor(neg_items)
        return user_id, pos_item_id, neg_item_id


class TextPointWiseAllFreezeDataset(Dataset):
    def __init__(self, user_history, train_pairs, item_num, item_embs, max_seq_len):
        super(TextPointWiseAllFreezeDataset, self).__init__()
        self.train_pairs = train_pairs
        self.item_num = item_num
        self.item_embs = item_embs
        self.max_seq_len = max_seq_len
        
        self.user_history = {}
        for uid in user_history.keys():
            self.user_history[uid] = np.array(user_history[uid])
        
    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, index):
        (user_id, item_id) = self.train_pairs[index]
        item_emb = self.item_embs[item_id]
        user_history = np.zeros(self.max_seq_len, dtype=np.int64)
        user_history_ = self.user_history[user_id]
        user_history_ = user_history_[user_history_ != item_id]
        user_history[:len(user_history_)] = user_history_
        return user_id, user_history, item_id, item_emb


class TextPointWiseFinetuneDataset(Dataset):
    def __init__(self, user_history, train_pairs, item_num, attention_mask, inputs_hidden_state, max_seq_len):
        super(TextPointWiseFinetuneDataset, self).__init__()
        self.train_pairs = train_pairs
        self.item_num = item_num
        self.attention_mask = attention_mask
        self.inputs_hidden_state = inputs_hidden_state
        self.max_seq_len = max_seq_len
        
        self.user_history = {}
        for uid in user_history.keys():
            self.user_history[uid] = np.array(user_history[uid])
        
    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, index):
        (user_id, item_id) = self.train_pairs[index]
        attention_mask = self.attention_mask[item_id]
        inputs_hidden_state = self.inputs_hidden_state[item_id]
        user_history = np.zeros(self.max_seq_len, dtype=np.int64)
        user_history_ = self.user_history[user_id]
        user_history_ = user_history_[user_history_ != item_id]
        user_history[:len(user_history_)] = user_history_
        return user_id, user_history, item_id, attention_mask, inputs_hidden_state
        

class IdPointWiseDataset(Dataset):
    def __init__(self, user_history, train_pairs, item_num, max_seq_len):
        super(IdPointWiseDataset, self).__init__()
        self.train_pairs = train_pairs
        self.item_num = item_num
        self.max_seq_len = max_seq_len
        
        self.user_history = {}
        for uid in user_history.keys():
            self.user_history[uid] = np.array(user_history[uid])
        
    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, index):
        (user_id, item_id) = self.train_pairs[index]
        user_history = np.zeros(self.max_seq_len, dtype=np.int64)
        user_history_ = self.user_history[user_id]
        user_history_ = user_history_[user_history_ != item_id]
        user_history[:len(user_history_)] = user_history_
        return user_id, user_history, item_id
    

class TextSeqAllFreezeTrainDataset(Dataset):
    def __init__(self, u2seq, item_embs, item_num, max_seq_len):
        self.behaviors = list(u2seq.items())
        self.item_embs = item_embs
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        user_id, seq = self.behaviors[idx] 
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items_id = torch.LongTensor([0] * mask_len_head + seq)
            
        item_emb = self.item_embs[sample_items_id]

        return sample_items_id, item_emb, torch.FloatTensor(log_mask)

class TextSeqFinetuneTrainDataset(Dataset):
    def __init__(self, u2seq, attention_mask, inputs_hidden_state, item_num, max_seq_len):
        self.behaviors = list(u2seq.items())
        self.attention_mask = attention_mask
        self.inputs_hidden_state = inputs_hidden_state
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        
    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        user_id, seq = self.behaviors[idx] 
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items_id = torch.LongTensor([0] * mask_len_head + seq)
            
        attention_mask = self.attention_mask[sample_items_id]
        inputs_hidden_state = self.inputs_hidden_state[sample_items_id]

        return sample_items_id, attention_mask, inputs_hidden_state, torch.FloatTensor(log_mask)


class IdSeqTrainDataset(Dataset):
    def __init__(self, u2seq, item_content, item_num, max_seq_len):
        self.behaviors = list(u2seq.items())
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        user_id, seq = self.behaviors[idx] 
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = torch.LongTensor([0] * mask_len_head + seq)
        sample_items_id = sample_items

        return sample_items_id, torch.LongTensor(sample_items), torch.FloatTensor(log_mask)


class SeqEvalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num):
        self.behaviors = list(u2seq.items())
        self.item_content = item_content
        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        user_id, seq = self.behaviors[idx] 
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_content[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            torch.FloatTensor(log_mask), \
            labels
            

class PointWiseEvalDataset(Dataset):
    def __init__(self, eval_pairs, user_content, item_num):
        self.eval_pairs = eval_pairs
        self.user_content = user_content
        self.item_num = item_num

    def __len__(self):
        return len(self.eval_pairs)

    def __getitem__(self, index):
        (user_id, target) = self.eval_pairs[index]
        user_emb = self.user_content[user_id]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), user_emb, labels


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
