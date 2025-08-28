from typing import Optional
from transformers import AutoConfig

def get_data_configs(dataset):
    uid_field = "user_id"
    iid_field = "item_id"
    item_text_field = "item_text"
    item_seq_field = "interactions"

    data_dir = f"../data/{dataset}/"

    if dataset in ["MIND_small", "MIND_large", "mind_200k", "mind_200k_20", "mind_200k_50", "mind_200k_200"]:
        inter_table = "behaviors"
        item_table = "news"

        old_uid_field = "userid"
        old_iid_field = "newsid"
        old_item_text_field = "title"
        old_item_seq_field = "behaviors"
    elif dataset in ["hm", "hm_100k", "hm_200k", "hm_200k_20", "hm_200k_50", "hm_200k_200"]:
        inter_table = "behaviors"
        item_table = "items"

        old_uid_field = "userid"
        old_iid_field = "itemid"
        old_item_text_field = "description"
        old_item_seq_field = "behaviors"
    elif dataset in ["bilibili", "bl_50k", "bl_200k", "bl_50k_20", "bl_50k_50", "bl_50k_200"]:
        inter_table = "behaviors"
        item_table = "videos"

        old_uid_field = "userid"
        old_iid_field = "videoid"
        old_item_text_field = "description"
        old_item_seq_field = "behaviors"
    elif dataset == "ks_200k":
        inter_table = "behaviors"
        item_table = "videos"

        old_uid_field = "userid"
        old_iid_field = "videoid"
        old_item_text_field = "description"
        old_item_seq_field = "behaviors"
    else:
        raise ValueError( f"dataset not supported: {dataset}")

    table_configs = {
        inter_table: {
            "filepath": data_dir + f"{inter_table}.tsv",
            "usecols": [old_uid_field, old_item_seq_field],
            "rename_cols": {
                old_uid_field: uid_field,
                old_item_seq_field: item_seq_field
            },
            "filed_type": {
                old_uid_field: str,
                old_item_seq_field: str
            },
            "token_seq_fields": [item_seq_field],
        },
        item_table: {
            "filepath": data_dir + f"{item_table}.tsv",
            "usecols": [old_iid_field, old_item_text_field],
            "filed_type": {
                "newsid": str,
                "title": str
            },
            "rename_cols": {
                old_iid_field: iid_field,
                old_item_text_field: item_text_field
            },
        },
    }

    data_configs = {
        "data_dir": data_dir,
        "uid_field": uid_field,
        "iid_field": iid_field,
        "item_text_field": item_text_field,
        "item_seq_field": item_seq_field,
        "inter_table": inter_table,
        "item_table": item_table,
    }

    data_configs.update({"table_configs": table_configs})

    return data_configs


class SeqRecDataModuleConfig:

    def __init__(self, dataset: str, **kwargs):
        self.dataset = dataset
        self.split_type = kwargs.get("split_type", "ratio")
        self.plm_name: str = kwargs.pop("plm_name", "facebook/opt-125m")
        self.plm_last_n_unfreeze: int = \
            kwargs.pop("plm_last_n_unfreeze", 0)
        self.min_item_seq_len: int = kwargs.pop("min_item_seq_len", 5)
        self.max_item_seq_len: Optional[int] = \
            kwargs.pop("max_item_seq_len", None)
        self.sasrec_seq_len: int = kwargs.pop("sasrec_seq_len", 20)
        self.tokenized_len: int = kwargs.pop("tokenized_len", 30)
        self.batch_size: int = kwargs.pop("batch_size", 64)
        self.num_workers: int = kwargs.pop("num_workers", 4)
        self.pin_memory: bool = kwargs.pop("pin_memory", True)

        plm_config = AutoConfig.from_pretrained(self.plm_name)
        
        
        if self.split_type not in ["ratio", "leave_one_out"]:
            raise ValueError("split_type must be in ['ratio', 'leave_one_out']")
        
        assert self.min_item_seq_len > 0
        assert self.tokenized_len > 0
        assert self.sasrec_seq_len > 0
        if self.max_item_seq_len is not None:
            assert self.max_item_seq_len > 0
            assert self.max_item_seq_len >= self.min_item_seq_len

        if self.plm_last_n_unfreeze is not None:
            assert self.plm_last_n_unfreeze >= -1
            assert self.plm_last_n_unfreeze <= plm_config.num_hidden_layers

class PreInferSeqRecDMConfig(SeqRecDataModuleConfig):
    def __init__(self, dataset: str, **kwargs):
        self.pre_inference_batch_size= kwargs.pop("pre_inference_batch_size", 1)
        self.pre_inference_precision = kwargs.pop("pre_inference_precision", 32)
        self.pre_inference_num_workers = kwargs.pop("pre_inference_num_workers", 4)
        self.pre_inference_layer_wise = kwargs.pop("pre_inference_layer_wise", False)
        self.pre_inference_devices= kwargs.pop(
            "pre_inference_devices", [0, 1, 2, 3, 4, 5, 6, 7]
            )
        super().__init__(dataset, **kwargs)
        if self.plm_last_n_unfreeze == -1:
            raise ValueError(
                f"The plm_last_n_unfreeze is -1, which means you want to fully fine-tune the PLM. "
                f"If you want to fully fine-tune the PLM, you should set --pre_inference to False."
                f"If you want to do pre-inference, please set --plm_last_n_unfreeze >= 0."
                )