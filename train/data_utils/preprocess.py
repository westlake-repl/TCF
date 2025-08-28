import numpy as np
import torch


def read_behaviors(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, file_logger, split_method='leave_one_out'):
    before_item_num = len(before_item_name_to_id)
    file_logger.info(f"item number before filtering: {before_item_num}")
    file_logger.info(f"for each user behavior, max_seq_len: {max_seq_len}, min_seq_len: {min_seq_len}")
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    file_logger.info('loading behaviors...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    file_logger.info(f"total interactions: {pairs_num}")

    file_logger.info('filtering items...')
    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    item_id_now_to_before = [0]
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_now_to_before.append(before_item_id)
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    item_id_now_to_before = torch.tensor(item_id_now_to_before, dtype=torch.long)
    file_logger.info("item number after filtering: {}".format(item_num))
    
    user_id = 1
    users_train, users_valid, users_test = {}, {}, {}
    train_pairs, valid_pairs, test_pairs = [], [], []
    train_item_counts = [0] * (item_num + 1)
    if split_method == 'leave_one_out':
        users_history_for_valid = {}
        users_history_for_test = {}
        for user_name, item_seqs in user_seq_dic.items():
            user_seq = [item_id_before_to_now[i] for i in item_seqs]
            train = user_seq[:-2]
            valid = user_seq[-(max_seq_len+2):-1]
            test = user_seq[-(max_seq_len+1):]
            users_train[user_id] = train
            users_valid[user_id] = valid
            users_test[user_id] = test
            
            for i in train:
                train_item_counts[i] += 1
                train_pairs.append((user_id, i))
            valid_pairs.append((user_id, valid[-1]))
            test_pairs.append((user_id, test[-1]))

            users_history_for_valid[user_id] = torch.LongTensor(np.array(train))
            users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
            user_id += 1
    elif split_method == 'ratio':
        # pass
        users_history = {}
        data = []
        for user_name, item_seqs in user_seq_dic.items(): 
            user_seq = [item_id_before_to_now[i] for i in item_seqs]
            data.append(user_seq[-(max_seq_len+1):])
            users_history[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
            user_id += 1
        users_history_for_test = users_history
        users_history_for_valid = users_history
        size = len(data)
        idx = np.random.permutation(size) 
        train_ratio, valid_ratio, test_ratio = 0.8, 0.1, 0.1
        train_idx = idx[:int(size*train_ratio)]
        valid_idx = idx[int(size*train_ratio): int(size*(train_ratio+valid_ratio))]
        test_idx = idx[int(size*(train_ratio+valid_ratio)):]
        idx = [train_idx, valid_idx, test_idx]
        users = [users_train, users_valid, users_test]
        stages = ['train', 'valid', 'test']
        for data_idx, user_dict, stage in zip(idx, users, stages):
            for idx_ in data_idx:
                user_id = idx_ + 1
                user_dict[user_id] = data[idx_]
                if stage == 'train':
                    for item_id in data[idx_]:
                        train_item_counts[item_id] += 1
                        train_pairs.append((user_id, item_id))
                elif stage == 'valid':
                    for item_id in data[user_id]:
                        valid_pairs.append((user_id, item_id))
                elif stage == 'test':
                    for item_id in data[user_id]:
                        test_pairs.append((user_id, item_id))
    else:
        raise NotImplementedError
    file_logger.info(f"behavior number after filtering, training:{len(users_train)} validation:{len(users_valid)} test:{len(users_test)}")
    file_logger.info(f"train point-wise interactions: {len(train_pairs)}")
    file_logger.info(f"valid point-wise interactions: {len(valid_pairs)}")
    file_logger.info(f"test point-wise interactions: {len(test_pairs)}")
    
    user_num = len(user_seq_dic)
    
    item_counts_powered = np.power(train_item_counts, 1)
    pop_prob_list = []
    for i in range(1, item_num + 1):
        pop_prob_list.append(item_counts_powered[i])
    pop_prob_list = pop_prob_list / sum(np.array(pop_prob_list))
    pop_prob_list = np.append([1], pop_prob_list)
    
    file_logger.info('prob max: {}, prob min: {}, prob mean: {}'.\
        format(max(pop_prob_list), min(pop_prob_list), np.mean(pop_prob_list)))
    file_logger.info(f'user behaviors num after filtering, train: {len(users_train)}, valid: {len(users_valid)}, test: {len(users_test)}')
    
    return user_num, item_num, item_id_to_dic, users_train, users_valid, users_test, pop_prob_list, \
           users_history_for_valid, users_history_for_test, item_name_to_id, item_id_now_to_before, \
              train_pairs, valid_pairs, test_pairs


def read_news(news_path):
    item_id_to_dic = {}
    item_id_to_name = {}
    item_name_to_id = {}
    item_id = 1
    with open(news_path, "r") as f:
        for line in f:
            doc_name, _ = line.strip('\n').split('\t')
            item_name_to_id[doc_name] = item_id
            item_id_to_dic[item_id] = doc_name
            item_id_to_name[item_id] = doc_name
            item_id += 1
    return item_id_to_dic, item_name_to_id, item_id_to_name


def read_news_bert(news_path, args, tokenizer):
    item_id_to_dic = {}
    item_id_to_name = {}
    item_name_to_id = {}
    item_id = 1
    with open(news_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            if len(splited) > 2:
                doc_name = splited[0]
                title = "\t".join(splited[1:])
            else:
                doc_name, title = splited
            title = tokenizer(title.lower(), max_length=args.num_words_title, padding='max_length', truncation=True)

            item_name_to_id[doc_name] = item_id
            item_id_to_name[item_id] = doc_name
            item_id_to_dic[item_id] = title
            item_id += 1
            
    return item_id_to_dic, item_name_to_id, item_id_to_name


def get_doc_input_bert(item_id_to_content, args):
    item_num = len(item_id_to_content) + 1

    news_title = np.zeros((item_num, args.num_words_title), dtype='int32')
    news_title_attmask = np.zeros((item_num, args.num_words_title), dtype='int32')

    for item_id in range(1, item_num):
        title = item_id_to_content[item_id]

        news_title[item_id] = title['input_ids']
        news_title_attmask[item_id] = title['attention_mask']

    return news_title, news_title_attmask
        


