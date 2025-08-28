import gc
from tqdm import tqdm
import torch.optim as optim
import re
import pprint
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer
from data_utils.dataset import (
    TextPointWiseAllFreezeDataset,
    TextPointWiseFinetuneDataset,
    IdPointWiseDataset,
    TextSeqAllFreezeTrainDataset,
    TextSeqFinetuneTrainDataset,
    IdSeqTrainDataset,
    TextSeqNegSampleAllFreezeTrainDataset,
    TextSeqNegSampleFinetuneTrainDataset,
    IdSeqNegSampleTrainDataset,
    TextPointWiseNegSampleAllFreezeDataset,
    TextPointWiseNegSampleFinetuneDataset,
    IdPointWiseNegSampleDataset,
    )

from parameters import parse_args
from model import SASRec, DSSM  
from data_utils import (
    read_news,
    read_news_bert,
    read_behaviors,
    eval_sasrec_model,
    eval_dssm_model,
    get_item_embeddings,
    get_user_embeddings,
)
from data_utils.utils import *
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.partial_opt import PartialOPTModel
from model.partial_bert import PartialBertModel
from model.partial_t5 import PartialT5EncoderModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PLM_ABBR = {
    "facebook/opt-125m": "OPT125M",
    "facebook/opt-350m": "OPT350M",
    "facebook/opt-1.3b": "OPT1.3B",
    "facebook/opt-2.7b": "OPT2.7B",
    "facebook/opt-6.7b": "OPT6.7B",
    "facebook/opt-13b": "OPT13B",
    "facebook/opt-30b": "OPT30B",
    "facebook/opt-66b": "OPT66B",
    "facebook/opt-175b": "OPT175B",
    "google/flan-t5-small": "T5SMALL",
    "google/flan-t5-base": "T5BASE",
    "google/flan-t5-large": "T5LARGE",
    "google/flan-t5-xl": "T5XL",
    "google/flan-t5-xxl": "T5XXL",
    "bert-base-uncased": "BERTBASE",
    "bert-large-uncased": "BERTLARGE",
}

PLM_N_LAYER = {
    "facebook/opt-125m": 12,
    "facebook/opt-350m": 24,
    "facebook/opt-1.3b": 24,
    "facebook/opt-2.7b": 32,
    "facebook/opt-6.7b": 32,
    "facebook/opt-13b": 40,
    "facebook/opt-30b": 48,
    "facebook/opt-66b": 64,
    "facebook/opt-175b": 96,
    "google/flan-t5-small": 8,
    "google/flan-t5-base": 12,
    "google/flan-t5-large": 24,
    "google/flan-t5-xl": 24,
    "google/flan-t5-xxl": 24,
    "bert-base-uncased": 12,
    "bert-large-uncased": 24,
}

PLM_EMB_DIM = {
    "facebook/opt-125m": 768,
    "facebook/opt-350m": 1024,
    "facebook/opt-1.3b": 2048,
    "facebook/opt-2.7b": 2560,
    "facebook/opt-6.7b": 4096,
    "facebook/opt-13b": 5120,
    "facebook/opt-30b": 7168,
    "facebook/opt-66b": 9216,
    "facebook/opt-175b": 12288,
    "google/flan-t5-small": 512,
    "google/flan-t5-base": 768,
    "google/flan-t5-large": 1024,
    "google/flan-t5-xl": 2048,
    "google/flan-t5-xxl": 4096,
    "bert-base-uncased": 768,
    "bert-large-uncased": 1024,
}


def parse_true_false(s):
    if s.lower() in [
        "true", "1", "yes", "y", "t", "on", "enable", "enabled", "ok"]:
        return True
    elif s.lower() in [
        "false", "0", "no", "n", "f", "off", "disable", "disabled", "nope"]:
        return False
    
    
def train(args, use_modal, local_rank):
    assert args.architecture in ["SASRec", "DSSM"]
    
    split_method = args.split_method
    architecture = args.architecture
    min_seq_len = args.min_seq_len
    max_seq_len = args.max_seq_len
    valid_start_epoch = args.valid_start_epoch
    
    use_warmup = parse_true_false(args.use_warmup)
    use_whitening = parse_true_false(args.use_whitening)
    use_pca = parse_true_false(args.use_pca)
    show_progress = parse_true_false(args.show_progress)
    
    
    file_logger.info(f"data splitting by {split_method} method")
    
    items_file_path = os.path.join(args.root_data_dir, args.dataset, args.news)
    behaviors_file_path = os.path.join(args.root_data_dir, args.dataset, args.behaviors)
    
    if use_modal:
        text_max_len = args.num_words_title
        language_model_name = args.language_model_name
        unfreeze_last_n_layer = args.unfreeze_last_n_layer
        keep_embed_layer = parse_true_false(args.keep_embedding_layer)
        plm_hidden_dim = PLM_EMB_DIM[language_model_name]
        tokenizer = AutoTokenizer.from_pretrained(language_model_name)

        file_logger.info("loading items...")
        (
            before_item_id_to_dic,
            before_item_name_to_id,
            before_item_id_to_name,
        ) = read_news_bert(items_file_path, args, tokenizer)
        
        file_logger.info("loading users' behaviors...")
        (
            user_num,
            item_num,
            item_id_to_dic,
            users_train,
            users_valid,
            users_test,
            pop_prob_list,
            users_history_for_valid,
            users_history_for_test,
            item_name_to_id,
            item_id_now_to_before,
            train_pairs,
            valid_pairs,
            test_pairs
        ) = read_behaviors(
            behaviors_file_path,
            before_item_id_to_dic,
            before_item_name_to_id,
            before_item_id_to_name,
            args.max_seq_len,
            args.min_seq_len,
            file_logger,
            split_method,
        )

        file_logger.info("combine news information...")
        news_title_attmask = np.zeros((item_num + 1, text_max_len), dtype=np.int32)
        for item_id in range(1, item_num + 1):
            news_title_attmask[item_id] = item_id_to_dic[item_id]['attention_mask']
       
        file_logger.info("loading pre-inferenced file...")
        inferenced_file_path = args.inference_path
        if PLM_ABBR[language_model_name] not in inferenced_file_path:
                raise ValueError(
                    f"The model:{language_model_name} mismatch the infernce file:{inferenced_file_path}"
                )
                
        if unfreeze_last_n_layer == PLM_N_LAYER[language_model_name] and keep_embed_layer:
            if not inferenced_file_path.endswith(".npy"):
                raise ValueError(
                            f"You must pre-inferenced the file in order to get the .npy item file."
                            "Unsupport file format:{inferenced_file_path.split('.')[-1]}")
        else:
            freeze_n_layer_str = f"freeze@{PLM_N_LAYER[language_model_name] - unfreeze_last_n_layer}"
            if freeze_n_layer_str not in inferenced_file_path:
                raise ValueError(
                    f"The unfreeze num of layers:{unfreeze_last_n_layer} " 
                    f"mismatch the infernce file:{inferenced_file_path}"
                    )
        
        file_logger.info("load plm model...")
        if unfreeze_last_n_layer > 0:
            if unfreeze_last_n_layer < PLM_N_LAYER[language_model_name]:
                if keep_embed_layer:
                    file_logger.info(
                        f"Warning: Although you set keep_embedding_layer to True, "
                         "it will be drop because you didn't freeze all layers.")
                    keep_embed_layer = False
                inputs_hidden_state = torch.load(inferenced_file_path)
                inputs_hidden_state = inputs_hidden_state[item_id_now_to_before]
                attention_mask = torch.tensor(news_title_attmask)
                item_content = attention_mask, inputs_hidden_state
            elif unfreeze_last_n_layer == PLM_N_LAYER[language_model_name]:
                if not keep_embed_layer:
                    inputs_hidden_state = torch.load(inferenced_file_path)
                    inputs_hidden_state = inputs_hidden_state[item_id_now_to_before]
                    attention_mask = torch.tensor(news_title_attmask)
                    item_content = attention_mask, inputs_hidden_state
                else:   
                    input_ids, attention_mask = np.load(inferenced_file_path, allow_pickle=True)
                    attention_mask = torch.tensor(news_title_attmask)
                    inputs_hidden_state = input_ids[item_id_now_to_before]
                    item_content = attention_mask, inputs_hidden_state
            else:
                raise ValueError( 
                    f"unfreeze_last_n_layer:{unfreeze_last_n_layer} "
                    f"should be less than PLM_N_LAYER:{PLM_N_LAYER[language_model_name]}"
                )
            file_logger.info("Successfully load the processed data for PLM model.")
            keep_hidden_layers_range = (-unfreeze_last_n_layer, None)

            if language_model_name.startswith("facebook/opt-"):
                text_model = PartialOPTModel.from_pretrained(
                    pretrained_model_name_or_path=language_model_name,
                    keep_embed_layer=keep_embed_layer,
                    keep_decoders_range=keep_hidden_layers_range)
            elif language_model_name.startswith("bert-"):
                text_model = PartialBertModel.from_pretrained(
                    pretrained_model_name_or_path=language_model_name,
                    keep_embed_layer=keep_embed_layer,
                    keep_encoders_range=keep_hidden_layers_range,
                    add_pooling_layer=False)
            elif language_model_name.startswith("google/flan-"):
                text_model = PartialT5EncoderModel.from_pretrained(
                    pretrained_model_name_or_path=language_model_name,
                    keep_embed_layer=keep_embed_layer,
                    keep_hidden_layers_range=keep_hidden_layers_range)
            else:
                raise ValueError(f"unsupported language model:{language_model_name}")
            file_logger.info("Successfully load the PLM model.")
                
        else:
            text_model = None
            if language_model_name.startswith("facebook/opt-"):
                mean_pooled_file_path = args.inference_path[:-3] + "_mean_pooled.pt"
                if os.path.exists(mean_pooled_file_path):
                    file_logger.info(f"load mean pooled hidden states from file:{mean_pooled_file_path}")
                    item_embs = torch.load(mean_pooled_file_path)
                else:
                    file_logger.info(f"mean pooling hidden states...")
                    inputs_hidden_state = torch.load(inferenced_file_path)
                    inputs_hidden_state = inputs_hidden_state[item_id_now_to_before]
                    attention_mask = torch.tensor(news_title_attmask)
                    
                    num_items = inputs_hidden_state.shape[0]
                    emb_dim = inputs_hidden_state.shape[-1]
                    item_embs = torch.empty((num_items, emb_dim), dtype=torch.float32)
                    data = tqdm(
                        zip(inputs_hidden_state, attention_mask),
                        total=num_items,
                        ncols=100,
                        desc="mean pooling",
                    ) if show_progress else zip(inputs_hidden_state, attention_mask)
                    for i, (hidden_state, mask) in enumerate(data):
                        mask_expanded = mask.unsqueeze(-1).expand_as(hidden_state)
                        item_embs[i] = torch.sum(hidden_state * mask_expanded,
                                                -2) / torch.clamp(
                                                    mask_expanded.sum(-2), min=1e-9)
                    # save the mean pooling result
                    torch.save(item_embs, mean_pooled_file_path)
                    
            elif language_model_name.startswith("bert-") or language_model_name.startswith("google/flan-"):
                inputs_hidden_state = torch.load(inferenced_file_path)
                inputs_hidden_state = inputs_hidden_state[item_id_now_to_before]
                attention_mask = torch.tensor(news_title_attmask)
                item_embs = inputs_hidden_state[:, 0, :]

            if use_pca:
                # whitening for item embeddings in torch tensor format
                file_logger.info("pca for item embeddings...")
                item_embs = item_embs.double().to(local_rank)
                new_dim = plm_hidden_dim
                # new_dim = args.embedding_dim
                mean = torch.mean(item_embs, dim=0)
                cov = torch.cov(item_embs.T)
                u, s, _ = torch.linalg.svd(cov)
                if use_whitening:
                    whiten_mat = torch.matmul(u, torch.diag(1. / torch.sqrt(s)))
                    item_embs = torch.matmul(item_embs - mean, whiten_mat[:, :new_dim])
                    file_logger.info(f"item embeddings shape after whitening: {item_embs.shape}")
                else:
                    item_embs = torch.matmul(item_embs - mean, u[:, :new_dim])
                    file_logger.info(f"item embeddings shape: {item_embs.shape}")
                item_embs = item_embs.float().cpu()
            
            item_content = item_embs
            
    else:
        file_logger.info("loading items...")
        (
            before_item_id_to_dic,
            before_item_name_to_id,
            before_item_id_to_name,
        ) = read_news(items_file_path)

        file_logger.info("loading users' behaviors...")
        (
            user_num,
            item_num,
            item_id_to_dic,
            users_train,
            users_valid,
            users_test,
            pop_prob_list,
            users_history_for_valid,
            users_history_for_test,
            item_name_to_id,
            item_id_now_to_before,
            train_pairs,
            valid_pairs,
            test_pairs
        ) = read_behaviors(
            behaviors_file_path,
            before_item_id_to_dic,
            before_item_name_to_id,
            before_item_id_to_name,
            args.max_seq_len,
            args.min_seq_len,
            file_logger,
            split_method,
        )
        item_content = np.arange(item_num + 1)
        text_model = None

    file_logger.info("build dataset...")
    if args.loss_type == "IBCE":
        if architecture == "SASRec":
            if use_modal:
                if text_model is not None:
                    train_dataset = TextSeqFinetuneTrainDataset(
                        u2seq=users_train,
                        inputs_hidden_state=inputs_hidden_state,
                        attention_mask=attention_mask,
                        item_num=item_num,
                        max_seq_len=max_seq_len,
                    )
                else:
                    train_dataset = TextSeqAllFreezeTrainDataset(
                        u2seq=users_train,
                        item_embs=item_embs,
                        item_num=item_num,
                        max_seq_len=max_seq_len,
                    )
            else:
                train_dataset = IdSeqTrainDataset(
                    u2seq=users_train,
                    item_content=item_content,
                    item_num=item_num,
                    max_seq_len=max_seq_len,
                )
        elif architecture == "DSSM":
            if use_modal:
                if text_model is not None:
                    train_dataset = TextPointWiseFinetuneDataset(
                        item_num=item_num,
                        train_pairs=train_pairs,
                        user_history=users_train,
                        attention_mask=attention_mask, 
                        inputs_hidden_state=inputs_hidden_state,
                        max_seq_len=max_seq_len,)
                else:
                    train_dataset = TextPointWiseAllFreezeDataset(
                        item_num=item_num,
                        train_pairs=train_pairs,
                        user_history=users_train,
                        item_embs=item_embs,
                        max_seq_len=max_seq_len,)
            else:
                print("neg sample")
                train_dataset = IdPointWiseDataset(
                    item_num=item_num,
                    train_pairs=train_pairs,
                    user_history=users_train,
                    max_seq_len=max_seq_len,)
                
    elif args.loss_type == "BCE":
        if architecture == "SASRec":
            if use_modal:
                if text_model is not None:
                    train_dataset = TextSeqNegSampleFinetuneTrainDataset(
                        u2seq=users_train,
                        inputs_hidden_state=inputs_hidden_state,
                        attention_mask=attention_mask,
                        item_num=item_num,
                        max_seq_len=max_seq_len,
                        neg_num=args.neg_num
                    )
                else:
                    train_dataset = TextSeqNegSampleAllFreezeTrainDataset(
                        u2seq=users_train,
                        item_embs=item_embs,
                        item_num=item_num,
                        max_seq_len=max_seq_len,
                        neg_num=args.neg_num
                    )
            else:
                train_dataset = IdSeqNegSampleTrainDataset(
                    u2seq=users_train,
                    item_content=item_content,
                    item_num=item_num,
                    max_seq_len=max_seq_len,
                    neg_num=args.neg_num
                )
        elif architecture == "DSSM":
            if use_modal:
                if text_model is not None:
                    train_dataset = TextPointWiseNegSampleFinetuneDataset(
                        item_num=item_num,
                        train_pairs=train_pairs,
                        user_history=users_train,
                        attention_mask=attention_mask, 
                        inputs_hidden_state=inputs_hidden_state,
                        neg_num=args.neg_num)
                else:
                    train_dataset = TextPointWiseNegSampleAllFreezeDataset(
                        item_num=item_num,
                        train_pairs=train_pairs,
                        user_history=users_train,
                        item_embs=item_embs,
                        neg_num=args.neg_num)
            else:
                train_dataset = IdPointWiseNegSampleDataset(
                    item_num=item_num,
                    train_pairs=train_pairs,
                    user_history=users_train,
                    neg_num=args.neg_num)
        
            
    file_logger.info("build DDP sampler...")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2**31
        worker_seed = initial_seed + worker_id + dist.get_rank()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    file_logger.info("build dataloader...")
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_reset_seed,
        pin_memory=True,
        sampler=train_sampler,
    )

    file_logger.info("build model...")
    if architecture == "SASRec":
        model = SASRec(args, pop_prob_list, item_num, use_modal, text_model).to(local_rank)
    elif architecture == "DSSM":
        model = DSSM(args, pop_prob_list, user_num, item_num, use_modal, text_model).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    if "None" not in args.load_ckpt_name:
        file_logger.info("load ckpt if not None...")
        ckpt_path = get_checkpoint(model_dir, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        file_logger.info("load checkpoint...")
        model.module.load_state_dict(checkpoint["model_state_dict"])
        file_logger.info(f"Model loaded from {ckpt_path}")
        start_epoch = int(re.split(r"[._-]", args.load_ckpt_name)[1])
        torch.set_rng_state(checkpoint["rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        is_early_stop = False
    else:
        checkpoint = None  # new
        ckpt_path = None  # new
        start_epoch = 0
        is_early_stop = True

    file_logger.info("model.cuda()...")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if use_modal:
        text_enc_params = []
        recsys_params = []
        for index, (name, param) in enumerate(model.module.named_parameters()):
            if param.requires_grad:
                if "text_encoder" in name:
                    text_enc_params.append(param)
                else:
                    recsys_params.append(param)
        
        if text_enc_params != []:
            optimizer = optim.AdamW([
                {
                    "params": text_enc_params,
                    "lr": args.fine_tune_lr,
                    "weight_decay": args.fine_tune_l2_weight,
                },
                {
                    "params": recsys_params,
                    "lr": args.lr,
                    "weight_decay": args.l2_weight,
                },
            ])
        else:
            optimizer = optim.AdamW(recsys_params, lr=args.lr, weight_decay=args.l2_weight)

        tot_n_params = sum(p.numel() for p in model.module.parameters())
        if text_model is not None:
            plm_n_params = sum(p.numel() for p in model.module.text_encoder.text_model.parameters())
            file_logger.info(f"{plm_n_params} parameters in plm:{PLM_ABBR[language_model_name]}, "
                             f"{tot_n_params} parameters in model")
        else:
            file_logger.info(f"{tot_n_params} parameters in model")
            
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        file_logger.info(f"{trainable_num} trainable parameters in model")

        for params in optimizer.param_groups:
            model_n_params = sum(p.numel() for p in params["params"])
            file_logger.info(f"{model_n_params} parameters in lr: {params['lr']}, weight_decay: {params['weight_decay']}")
    else:
        optimizer = optim.AdamW(model.module.parameters(),
                                lr=args.lr,
                                weight_decay=args.l2_weight)
    
    total_training_steps = len(users_train) * args.epoch / args.batch_size / dist.get_world_size()
    total_training_steps = math.ceil(total_training_steps)
    total_scheduler_steps = total_training_steps * 1 // 5
    warmup_steps = int(total_scheduler_steps * 0.1)
    
    file_logger.info(f"total steps: {total_training_steps}")
    file_logger.info(f"total scheduler steps: {total_scheduler_steps}")
    file_logger.info(f"warmup steps: {warmup_steps}")
     
    def lr_lambda_with_linear_warmup_log_annealing(current_step, warmup_steps, total_scheduler_steps, final_ratio=0.1):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            if current_step > total_scheduler_steps:
                return final_ratio
            else:
                return math.exp(math.log(final_ratio) * (current_step - warmup_steps) / (total_scheduler_steps - warmup_steps))
        
    def lr_lambda_constant(current_step):
        return 1.0
    
    if use_warmup:
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=[
                lambda current_step: lr_lambda_with_linear_warmup_log_annealing(
                    current_step=current_step,
                    warmup_steps=warmup_steps,
                    total_scheduler_steps=total_scheduler_steps,
                ),
                lr_lambda_constant,
            ],
            last_epoch=-1)
    else:
        lr_scheduler = None

    if "None" not in args.load_ckpt_name:
        optimizer.load_state_dict(checkpoint["optimizer"])
        file_logger.info(f"optimizer loaded from {ckpt_path}")

    file_logger.info("\n")
    file_logger.info("Training...")
    file_logger.info(str(model.module))
    next_set_start_time = time.time()
    best_epoch, early_stop_epoch = 0, args.epoch
    max_eval_hit10, max_eval_ndcg10, early_stop_count = 0, 0, 0

    steps_for_log, _ = para_and_log(
        model,
        len(users_train),
        args.batch_size,
        file_logger,
        logging_num=args.logging_num,
        testing_num=args.testing_num,
    )
    scaler = torch.cuda.amp.GradScaler()
    if "None" not in args.load_ckpt_name:
        scaler.load_state_dict(checkpoint["scaler_state"])
        file_logger.info(f"scaler loaded from {ckpt_path}")
    Log_screen.info("{} train start".format(args.label_screen))

    every_n_epoch_eval = args.every_n_epoch_eval
    early_stop_patience = args.early_stop_patience

    best_path = None
    
    for ep in range(args.epoch):
        now_epoch = start_epoch + ep + 1
        file_logger.info("\n")
        file_logger.info("epoch {} start".format(now_epoch))
        file_logger.info("")
        loss, batch_index, need_break = 0.0, 1, False
        
        if text_model is not None and PLM_ABBR[language_model_name] == "OPT66B":
            gc.collect()
            torch.cuda.empty_cache()
        
        model.train()
        train_dl.sampler.set_epoch(now_epoch)

        dl = tqdm(train_dl,
                  total=len(train_dl),
                  ncols=100,
                  desc=f"Train {ep:>5}") \
            if show_progress else train_dl
        
        if lr_scheduler is not None:
            file_logger.info('start of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_last_lr()))
        
        for data in dl:
            if args.loss_type == "IBCE":
                if architecture == "SASRec":
                    if use_modal:
                        if text_model is not None:
                            sample_items_id, attention_mask, inputs_hidden_state, log_mask = data 
                            sample_items_id, attention_mask, inputs_hidden_state, log_mask = \
                                sample_items_id.to(local_rank), attention_mask.to(local_rank), inputs_hidden_state.to(local_rank), log_mask.to(local_rank)
                            attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
                            inputs_hidden_state = inputs_hidden_state.reshape(-1, *inputs_hidden_state.shape[-2:])
                            sample_items = attention_mask, inputs_hidden_state
                        else:
                            sample_items_id, sample_items, log_mask = data
                            sample_items_id, sample_items, log_mask = \
                                sample_items_id.to(local_rank), sample_items.to(local_rank), log_mask.to(local_rank)
                            sample_items = sample_items.reshape(-1, sample_items.shape[-1])
                    else:
                        sample_items_id, sample_items, log_mask = data
                        sample_items_id, sample_items, log_mask = \
                            sample_items_id.to(local_rank), sample_items.to(local_rank), log_mask.to(local_rank)
                        sample_items = sample_items.view(-1)
                    sample_items_id = sample_items_id.view(-1)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        bz_loss = model(sample_items_id, sample_items, log_mask, local_rank)
                
                elif architecture == "DSSM":
                    if use_modal:
                        if text_model is not None:
                            user_ids, user_history, item_ids, attention_mask, inputs_hidden_state = data
                            user_ids, user_history, item_ids, attention_mask, inputs_hidden_state = \
                                user_ids.to(local_rank), user_history.to(local_rank), item_ids.to(local_rank), attention_mask.to(local_rank), inputs_hidden_state.to(local_rank)
                            attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
                            inputs_hidden_state = inputs_hidden_state.reshape(-1, *inputs_hidden_state.shape[-2:])
                            sample_items = item_ids, attention_mask, inputs_hidden_state
                        else:
                            user_ids, user_history, item_ids, sample_items = data
                            user_ids, user_history, item_ids, sample_items = \
                                user_ids.to(local_rank), user_history.to(local_rank), item_ids.to(local_rank), sample_items.to(local_rank)
                            sample_items = sample_items.reshape(-1, sample_items.shape[-1])
                            sample_items = item_ids, sample_items
                    else:
                        user_ids, user_history, sample_items = data
                        user_ids, user_history, sample_items = \
                            user_ids.to(local_rank), user_history.to(local_rank), sample_items.to(local_rank)
                            
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        bz_loss = model(user_ids, user_history, sample_items, local_rank)
                        
            elif args.loss_type == "BCE":
                if architecture == "SASRec":
                    if use_modal:
                        if text_model is not None:
                            pos_item_id, neg_item_id, pos_attention_mask, neg_attention_mask, pos_inputs_hidden_state, neg_inputs_hidden_state, mask = data 
                            pos_item_id, neg_item_id, pos_attention_mask, neg_attention_mask, pos_inputs_hidden_state, neg_inputs_hidden_state, mask = \
                                pos_item_id.to(local_rank), neg_item_id.to(local_rank), pos_attention_mask.to(local_rank), neg_attention_mask.to(local_rank), pos_inputs_hidden_state.to(local_rank), neg_inputs_hidden_state.to(local_rank), mask.to(local_rank)
                            sample_items_id = pos_item_id, neg_item_id
                            pos_attention_mask = pos_attention_mask.reshape(-1, pos_attention_mask.shape[-1])
                            pos_inputs_hidden_state = pos_inputs_hidden_state.reshape(-1, *pos_inputs_hidden_state.shape[-2:])
                            neg_attention_mask = neg_attention_mask.reshape(-1, neg_attention_mask.shape[-1])
                            neg_inputs_hidden_state = neg_inputs_hidden_state.reshape(-1, *neg_inputs_hidden_state.shape[-2:])
                            sample_items = (pos_attention_mask, pos_inputs_hidden_state), (neg_attention_mask, neg_inputs_hidden_state)
                        else:
                            pos_item_id, neg_item_id, pos_item_emb, neg_item_emb, mask = data
                            pos_item_id, neg_item_id, pos_item_emb, neg_item_emb, mask = \
                                pos_item_id.to(local_rank), neg_item_id.to(local_rank), pos_item_emb.to(local_rank), neg_item_emb.to(local_rank), mask.to(local_rank)
                            sample_items_id = pos_item_id, neg_item_id
                            pos_item_emb = pos_item_emb.reshape(-1, pos_item_emb.shape[-1])
                            neg_item_emb = neg_item_emb.reshape(-1, neg_item_emb.shape[-1])
                            sample_items = pos_item_emb, neg_item_emb
                    else:
                        pos_item_id, neg_item_id, mask = data
                        pos_item_id, neg_item_id, mask = \
                            pos_item_id.to(local_rank ), neg_item_id.to(local_rank), mask.to(local_rank)
                        pos_item_id = pos_item_id.view(-1)
                        neg_item_id = neg_item_id.view(-1)
                        sample_items_id = pos_item_id, neg_item_id
                        sample_items = sample_items_id

                    log_mask = mask
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        bz_loss = model(sample_items_id, sample_items, log_mask, local_rank)
                
                elif architecture == "DSSM":
                    if use_modal:
                        if text_model is not None:
                            user_ids, pos_attention_mask, pos_inputs_hidden_state, neg_attention_mask, neg_inputs_hidden_state = data
                            user_ids, pos_attention_mask, pos_inputs_hidden_state, neg_attention_mask, neg_inputs_hidden_state = \
                                user_ids.to(local_rank), pos_attention_mask.to(local_rank), pos_inputs_hidden_state.to(local_rank), neg_attention_mask.to(local_rank), neg_inputs_hidden_state.to(local_rank)
                            pos_attention_mask = pos_attention_mask.reshape(-1, pos_attention_mask.shape[-1])
                            neg_attention_mask = neg_attention_mask.reshape(-1, neg_attention_mask.shape[-1])
                            pos_inputs_hidden_state = pos_inputs_hidden_state.reshape(-1, *pos_inputs_hidden_state.shape[-2:])
                            neg_inputs_hidden_state = neg_inputs_hidden_state.reshape(-1, *neg_inputs_hidden_state.shape[-2:])
                            sample_items = (pos_attention_mask, pos_inputs_hidden_state), (neg_attention_mask, neg_inputs_hidden_state)
                            user_history = None
                        else:
                            user_ids, pos_item_embs, neg_item_embs = data
                            user_ids, pos_item_embs, neg_item_embs = \
                                user_ids.to(local_rank), pos_item_embs.to(local_rank), neg_item_embs.to(local_rank)
                            pos_item_embs = pos_item_embs.reshape(-1, pos_item_embs.shape[-1])
                            neg_item_embs = neg_item_embs.reshape(-1, neg_item_embs.shape[-1])
                            sample_items = pos_item_embs, neg_item_embs
                            user_history = None
                    else:
                        user_ids, pos_item_ids, neg_item_ids = data
                        user_ids, pos_item_ids, neg_item_ids = \
                            user_ids.to(local_rank), pos_item_ids.to(local_rank), neg_item_ids.to(local_rank)
                        sample_items = pos_item_ids, neg_item_ids
                        user_history = None
                        
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        bz_loss = model(user_ids, user_history, sample_items, local_rank)

            loss += bz_loss.data.float()
            scaler.scale(bz_loss).backward()
            if use_modal:
                torch.nn.utils.clip_grad_norm_(
                    model.module.text_encoder.parameters(), max_norm=1.0, norm_type=2)
            
            scaler.step(optimizer)
            # check nan https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            
            if lr_scheduler is not None:
                if not skip_lr_sched:
                    lr_scheduler.step()

            if torch.isnan(loss.data):
                need_break = True
                break

            if batch_index % steps_for_log == 0:
                file_logger.debug(
                    "cnt: {}, Ed: {}, batch loss: {:.5f}, sum loss: {:.5f}".
                    format(
                        batch_index,
                        batch_index * args.batch_size,
                        loss.data / batch_index,
                        loss.data,
                    ))
            batch_index += 1

        if lr_scheduler is not None:
            file_logger.info('end of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_last_lr()))
        
        if text_model is not None and PLM_ABBR[language_model_name] == "OPT66B":
            gc.collect()
            torch.cuda.empty_cache()
         
        if not need_break and now_epoch % every_n_epoch_eval == 0 and now_epoch >= valid_start_epoch:
            file_logger.info("")
            (
                max_eval_hit10,
                max_eval_ndcg10,
                best_epoch,
                early_stop_epoch,
                early_stop_count,
                need_break,
                need_save,
            ) = run_eval(
                now_epoch,
                best_epoch,
                early_stop_epoch,
                max_eval_hit10,
                max_eval_ndcg10,
                early_stop_count,
                model,
                item_content,
                users_history_for_valid,
                users_valid,
                2048,
                user_num,
                item_num,
                use_modal,
                args.mode,
                is_early_stop,
                local_rank,
                early_stop_patience,
                valid_pairs,
                args
            )
            
            model.train()

            if best_epoch == now_epoch and dist.get_rank() == 0:
                if best_path is not None:
                    os.remove(best_path)
                best_path = save_model(
                    now_epoch,
                    model,
                    model_dir,
                    optimizer,
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(),
                    scaler,
                    file_logger,
                )
        
        file_logger.info("")
        next_set_start_time = report_time_train(batch_index, now_epoch, loss,
                                                next_set_start_time,
                                                start_time, file_logger)
        Log_screen.info("{} training: epoch {}/{}".format(
            args.label_screen, now_epoch, args.epoch))
        if need_break:
            if now_epoch == 1:
                record_header = "dataset,arch,is_modal,plm_name,emb_size,bs,lr,wd,dp,plm_lr,plm_wd,total_epoch,best_epoch,stop_epoch,valid_hit10,test_hit10,test_ndcg10,num_blocks,num_heads"
                plm_name = "NONE" if args.item_tower != "modal" else PLM_ABBR[language_model_name]
                plm_wd = float('nan') if args.item_tower != "modal" else args.fine_tune_l2_weight
                plm_lr = float('nan') if args.item_tower != "modal" else args.fine_tune_lr
                record_line = f"{args.dataset},{architecture},{args.item_tower},{plm_name},{args.embedding_dim}," \
                            + f"{args.batch_size},{args.lr},{args.l2_weight},{args.drop_rate},{plm_lr},{plm_wd},{args.epoch},{1},{1}," \
                            + f"{float('nan')},{float('nan')},{float('nan')},{args.transformer_block},{args.num_attention_heads}"
                file_logger.info(record_header)
                file_logger.info(record_line)
            break
        

    file_logger.info("\n")
    file_logger.info("%" * 90)
    file_logger.info(" max eval Hit10 {:0.5f}  in epoch {}".format(
        max_eval_hit10 * 100, best_epoch))
    file_logger.info(" early stop in epoch {}".format(early_stop_epoch))
    file_logger.info("the End")
    Log_screen.info("{} train end in epoch {}".format(args.label_screen,
                                                      early_stop_epoch))
    
    valid_best_hit10 = max_eval_hit10 * 100
    
    checkpoint = torch.load(best_path, map_location=torch.device("cpu"))
    file_logger.info("load checkpoint from best: {}".format(best_path))
    model.module.load_state_dict(checkpoint["model_state_dict"])
    file_logger.info(f"Model loaded from {best_path}")
    torch.set_rng_state(checkpoint["rng_state"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    
    if text_model is not None and PLM_ABBR[language_model_name] == "OPT66B":
        gc.collect()
        torch.cuda.empty_cache()
        
    max_eval_hit10, best_epoch_, early_stop_count = 0, 0, 0
    max_eval_hit10, max_eval_ndcg10, _, _, _, _, _ = run_eval(
        now_epoch,
        best_epoch_,
        early_stop_epoch,
        max_eval_hit10,
        max_eval_ndcg10,
        early_stop_count,
        model,
        item_content,
        users_history_for_test,
        users_test,
        2048,
        user_num,
        item_num,
        use_modal,
        'test',
        is_early_stop,
        local_rank,
        early_stop_patience,
        test_pairs,
        args,
    )
    file_logger.info("test Hit10 {:0.5f}".format(max_eval_hit10 * 100))
    
    test_hit10, test_ndcg10 = max_eval_hit10 * 100, max_eval_ndcg10 * 100
    record_header = "dataset,arch,is_modal,plm_name,finetune_last,emb_size,bs,lr,wd,dp,plm_lr,plm_wd,total_epoch,best_epoch,stop_epoch,valid_hit10,test_hit10,test_ndcg10,num_blocks,num_heads"
    plm_name = "NONE" if args.item_tower != "modal" else PLM_ABBR[language_model_name]
    finetune_layer = "NONE" if args.item_tower != "modal" else unfreeze_last_n_layer
    plm_wd = float('nan') if args.item_tower != "modal" else args.fine_tune_l2_weight
    plm_lr = float('nan') if args.item_tower != "modal" else args.fine_tune_lr
    record_line = f"{args.dataset},{architecture},{args.item_tower},{plm_name},{finetune_layer},{args.embedding_dim}," \
                + f"{args.batch_size},{args.lr},{args.l2_weight},{args.drop_rate},{plm_lr},{plm_wd},{args.epoch},{best_epoch},{early_stop_epoch}," \
                + f"{valid_best_hit10:.5f},{test_hit10:.5f},{test_ndcg10:.5f},{args.transformer_block},{args.num_attention_heads}"
    file_logger.info(record_header)
    file_logger.info(record_line)
    
    
def run_eval(
    now_epoch,
    best_epoch,
    early_stop_epoch,
    max_eval_hit10,
    max_eval_ndcg10,
    early_stop_count,
    model,
    item_content,
    user_history,
    users_eval,
    batch_size,
    user_num,
    item_num,
    use_modal,
    mode,
    is_early_stop,
    local_rank,
    early_stop_patience,
    eval_pairs,
    args,
):
    eval_start_time = time.time()
    file_logger.info("Validating...")
    item_embeddings = get_item_embeddings(model, item_content, batch_size,
                                          args, use_modal, local_rank)

    if args.architecture == 'SASRec':
        valid_hit10, valid_ndcg10 = eval_sasrec_model(
            model,
            user_history,
            users_eval,
            item_embeddings,
            batch_size,
            args,
            item_num,
            file_logger,
            mode,
            local_rank,
        )
        
    elif args.architecture == 'DSSM':
        user_embeddings = get_user_embeddings(model, user_num, batch_size, args, local_rank)
        valid_hit10, valid_ndcg10 = eval_dssm_model(
            model,
            user_history,
            eval_pairs,
            user_embeddings,
            item_embeddings,
            batch_size,
            args,
            item_num,
            file_logger,
            mode,
            local_rank,
        )

        
    report_time_eval(eval_start_time, file_logger)
    file_logger.info("")
    need_break = False
    need_save = False
    if valid_hit10 > max_eval_hit10:
        max_eval_hit10 = valid_hit10
        max_eval_ndcg10 = valid_ndcg10
        best_epoch = now_epoch
        early_stop_count = 0
        need_save = False
    else:
        early_stop_count += 1
        if early_stop_count > early_stop_patience:
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch
    return (
        max_eval_hit10,
        max_eval_ndcg10,
        best_epoch,
        early_stop_epoch,
        early_stop_count,
        need_break,
        need_save,
    )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    local_rank = args.local_rank
    language_model_name = args.language_model_name
    args.word_embedding_dim = PLM_EMB_DIM[language_model_name]
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    setup_seed(12345)
    gpus = torch.cuda.device_count()

    if "modal" in args.item_tower:
        is_use_modal = True
        model_load = PLM_ABBR[args.language_model_name]
        dir_label = (str(args.item_tower) +
                     f"_{model_load}_freeze_{args.freeze_paras_before}")
    else:
        is_use_modal = False
        model_load = "id"
        dir_label = str(args.item_tower)

    batch_size = args.batch_size * gpus
    log_paras = (f"{model_load}_ed_{args.embedding_dim}"
                 f"_bs_{batch_size}_lr_{args.lr}_Flr_{args.fine_tune_lr}"
                 f"_L2_{args.l2_weight}_FL2_{args.fine_tune_l2_weight}")
    model_dir = os.path.join("./checkpoint_" + dir_label, "cpt_" + log_paras)
    time_run = time.strftime("-%Y%m%d-%H%M%S", time.localtime())
    args.label_screen = args.label_screen + time_run

    file_logger, Log_screen = setuplogger(dir_label,
                                       log_paras, time_run, args.mode,
                                       dist.get_rank(), args.behaviors)
    
    args_info = pprint.pformat(vars(args))
    file_logger.info(args_info)
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    if "train" in args.mode:
        train(args, is_use_modal, local_rank)
    end_time = time.time()
    h, m, s = get_time(start_time, end_time)
    file_logger.info(
        f"##### (time) all: {h} hours {m} minutes {s} seconds #####")
