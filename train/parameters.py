from data_utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    # ============== data_dir ==============
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--item_tower", type=str, default="id")
    parser.add_argument("--root_data_dir", type=str, default="../data/",)
    parser.add_argument("--dataset", type=str, default='bl_50k')
    parser.add_argument("--behaviors", type=str, default='behaviors.tsv')
    parser.add_argument("--split_method", type=str, default='leave_one_out')
    parser.add_argument("--cold_file", type=str, default='None')
    parser.add_argument("--new_file", type=str, default='None')
    parser.add_argument("--news", type=str, default='news.tsv')
    parser.add_argument("--use_pca", type=str, default=False)
    parser.add_argument("--use_whitening", type=str, default='True')
    parser.add_argument("--mlm_mean_pooling", type=str, default='False')

    # ============== train parameters ==============
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--valid_start_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--l2_weight", type=float, default=0)
    parser.add_argument("--fine_tune_l2_weight", type=float, default=0)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="IBCE")
    parser.add_argument("--power", type=float, default=1)
    parser.add_argument("--use_warmup", type=str, default="False")
    parser.add_argument("--neg_num", type=int, default=1)
    parser.add_argument("--n_mlp_layers", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--every_n_epoch_eval", type=int, default=1)
    parser.add_argument("--valid_data_num", type=int, default=0) # 0 means all
    parser.add_argument("--test_data_num", type=int, default=0) # 0 means all

    # ============== model parameters ==============
    parser.add_argument("--architecture", type=str, default="SASRec")
    parser.add_argument("--language_model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--inference_path", type=str, default="../data/hm_200k/hm_200k_maxlen@INF_minlen@5_toklen@30_saslen@20_processed/items_OPT125M.processed.npy")
    parser.add_argument("--freeze_paras_before", type=int, default=165)
    parser.add_argument("--n_project_out_mlp_layers", type=int, default=1)
    parser.add_argument("--word_embedding_dim", type=int, default=768)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--transformer_block", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--min_seq_len", type=int, default=5)
    parser.add_argument("--unfreeze_last_n_layer", type=int, default=0)
    parser.add_argument("--keep_embedding_layer", type=str, default='False')

    # ============== switch and logging setting ==============
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--load_ckpt_name", type=str, default='None')
    parser.add_argument("--label_screen", type=str, default='None')
    parser.add_argument("--logging_num", type=int, default=8)
    parser.add_argument("--testing_num", type=int, default=1)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--show_progress", type=str, default='True')
                        
    # ============== news information==============
    parser.add_argument("--num_words_title", type=int, default=30)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
