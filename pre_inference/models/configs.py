class SeqRecConfig:

    def __init__(self, item_token_num: int, **kwargs):
        self.item_token_num = item_token_num
        self.lr = kwargs.pop("lr", 1e-3)
        self.weight_decay = kwargs.pop("weight_decay", 0.1)
        self.sasrec_seq_len = kwargs.pop("sasrec_seq_len", 20)
        self.sasrec_n_layers = kwargs.pop("sasrec_n_layers", 2)
        self.sasrec_n_heads = kwargs.pop("sasrec_n_heads", 2)
        self.sasrec_hidden_size = kwargs.pop("sasrec_hidden_size", 64)
        self.sasrec_inner_size = kwargs.pop("sasrec_inner_size", 256)
        self.sasrec_hidden_dropout = kwargs.pop("sasrec_hidden_dropout", 0.5)
        self.sasrec_attention_dropout = kwargs.pop("sasrec_attention_dropout",
                                                   0.5)
        self.layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-12)
        self.initializer_range = kwargs.pop("initializer_range", 0.02)
        self.topk_list = kwargs.pop("topk_list", [5, 10, 20])

        if kwargs:
            raise ValueError(f'Unrecognized arguments: {kwargs}')


class TextSeqRecConfig(SeqRecConfig):

    def __init__(self, item_token_num: int, **kwargs):
        self.plm_name = kwargs.pop("plm_name", 'facebook/opt-125m')
        self.plm_last_n_unfreeze = kwargs.pop("plm_last_n_unfreeze", 0)

        plm_lr = kwargs.pop("plm_lr", 1e-5)
        plm_lr_layer_decay = kwargs.pop("plm_lr_layer_decay", 0.8)
        plm_weight_decay = kwargs.pop("plm_weiget_decay", 0.0)
        if self.plm_last_n_unfreeze != 0:
            self.plm_lr = plm_lr
            self.plm_lr_layer_decay = plm_lr_layer_decay
            self.plm_weight_decay = plm_weight_decay

        self.projection_n_layers = kwargs.pop("projection_n_layers", 5)
        self.projection_inner_sizes = kwargs.pop("projection_inner_sizes",
                                                 [3136, 784, 3136])
        assert len(self.projection_inner_sizes) == self.projection_n_layers - 1

        super().__init__(item_token_num, **kwargs)


class OPTSeqRecConfig(TextSeqRecConfig):

    def __init__(self,
                 item_token_num: int,
                 pooling_method: str = 'mean',
                 **kwargs):
        self.pooling_method = pooling_method

        if self.pooling_method not in ['mean', 'last']:
            raise ValueError(
                f"pooling_method {self.pooling_method} is not supported.")

        super().__init__(item_token_num, **kwargs)


class BERTSeqRecConfig(TextSeqRecConfig):

    def __init__(self,
                 item_token_num: int,
                 pooling_method: str = 'cls',
                 **kwargs):
        self.pooling_method = pooling_method

        if self.pooling_method not in ['cls', 'mean', 'pooler']:
            raise ValueError(
                f"pooling_method {self.pooling_method} is not supported.")

        super().__init__(item_token_num, **kwargs)
