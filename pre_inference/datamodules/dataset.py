import numpy as np
from torch.utils.data import Dataset
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class IDSeqRecDataset(Dataset):
    def __init__(
        self,
        input_id_seqs,
        target_id_seqs,
        padding_idx=0,
    ):
        self._len = len(input_id_seqs)
        self.padding_idx = padding_idx
        self.input_id_seqs = input_id_seqs
        self.target_id_seqs = target_id_seqs
        self.item_seq_masks = self._get_masks(self.input_id_seqs)

    def _get_masks(self, data):
        masks = np.where(data != self.padding_idx, True, False)
        return masks
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        item_id_seq = self.input_id_seqs[idx]
        target_id_seq = self.target_id_seqs[idx]
        item_seq_mask = self.item_seq_masks[idx]
        return target_id_seq, item_id_seq, item_seq_mask


class TextSeqRecDataset(IDSeqRecDataset):

    def __init__(
        self,
        input_id_seqs,
        target_id_seqs,
        tokenized_ids,
        attention_mask,
        padding_idx=0,
    ):
        super().__init__(input_id_seqs, target_id_seqs, padding_idx)
        self.tokenized_ids = tokenized_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        target_id_seq, item_id_seq, item_seq_mask = super().__getitem__(idx)
        tokenized_ids = self.tokenized_ids[item_id_seq]
        attention_mask = self.attention_mask[item_id_seq]
        return target_id_seq, item_id_seq, item_seq_mask, tokenized_ids, attention_mask
