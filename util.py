import argparse
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm
import torch
import dgl
import os
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sentence_transformers import SentenceTransformer
#import matplotlib.pyplot as plt
import math
import re
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument(
        '--network_files', type=list,
        default=[
            # '../data/network/202008-network.lj',
            # '../data/network/202009-network.lj',
            'data/network/202010-network.lj'
        ]
    )
    parser.add_argument(
        '--text_files', type=list,
        default=[
            # '../data/text/202008-text.lj',
            # '../data/text/202009-text.lj',
            'data/text/202010-text.lj'
        ]
    )
    parser.add_argument(
        '--user_files', type=list,
        default=[
            # '../data/text/202008-text.lj',
            # '../data/text/202009-text.lj',
            'data/user/user_info.txt'
        ]
    )
    parser.add_argument('--hashtag_file', type=str, default='data/hashtags.csv')
    parser.add_argument('--label_threshold', type=float, default=0.6)
    parser.add_argument('--valid_size', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='bert')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--random_walk_length', type=int, default=2)
    parser.add_argument('--num_random_walks', type=int, default=10)
    parser.add_argument('--num_neighbors', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)


    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_norm', type=float, default=1.)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--logging_dir', type=str, default='logs')
    parser.add_argument('--model', default='new', choices=['gcn', 'sage', 'pinsage', 'new', 'gin', 'gat', 'rgcn', 'rgat'])

    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def load_text_data(text_files, hashtag_file):
    if not os.path.exists('data/text/text_data.pickle'):
        label_hashtags = list(pd.read_csv(hashtag_file)['hashtag'].apply(lambda x: x.lower()).values)
        texts = []

        for text_file in text_files:
            with open(text_file, 'r', encoding='utf-8') as f:
                for i, line in tqdm(enumerate(f)):
                    data = eval(line)
                    text = data['text']
                    text = text.replace('#', ' #')
                    text = text.strip() + ' '
                    for hashtag in data['hashtags']:
                        if hashtag['text'].lower() in label_hashtags:
                            text = text.replace('#' + hashtag['text'] + ' ', ' ')
                    text = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', 'URL', text)
                    text = re.sub(r'\s+', ' ', text)
                    text = text.strip()
                    texts.append(text)

        pickle.dump(texts, open('data/text/text_data.pickle', 'wb'))
    else:
        texts = pickle.load(open('data/text/text_data.pickle', 'rb'))
    return texts


def build_graph(network_files, text_files, user_files):
    if os.path.exists('data/network/hete_g.pickle'):
        g = pickle.load(open('data/network/hete_g.pickle', 'rb'))
        u_l_ratio = pickle.load(open('data/user/user_label_ratio.pickle', 'rb'))
    else:
        data_dict = defaultdict(list)
        t_labels = {}
        tids = []
        print("reading texts...")
        for text_file in text_files:
            with open(text_file, 'r', encoding='utf-8') as f:
                for i, line in tqdm(enumerate(f)):
                    data = eval(line)
                    tid = data['tweet_id']
                    t_labels[tid] = data['label']
                    tids.append(tid)
        tid_set = set(tids)

        uids = []
        for user_file in user_files:
            with open(user_file, 'r', encoding='utf-8') as f:
                for i, line in tqdm(enumerate(f)):
                    data = eval(line)
                    user_id = data['id_str']
                    uids.append(user_id)
        uid_set = set(uids)

        print("building graphs...")
        t_mapping = {id: idx for idx, id in enumerate(tids)}  
        u_mapping = {id: idx for idx, id in enumerate(uids)}  
        u_t_labels = {}
        for network_file in network_files:
            with open(network_file, 'r', encoding='utf-8') as f:
                for i, line in tqdm(enumerate(f)):
                    data = eval(line)
                    if 'r_tid' in data:  # retweet
                        r_tid = data['r_tid']
                        if r_tid in tid_set:
                            uid = data['uid']
                            if uid in uid_set: 
                                t_idx = t_mapping[r_tid]
                                u_idx = u_mapping[uid]
                                if uid not in u_t_labels:
                                    u_t_labels[uid] = []
                                if t_labels[r_tid] == 'JB':
                                    u_t_labels[uid].append(0)
                                elif t_labels[r_tid] == 'DT':
                                    u_t_labels[uid].append(1)
                            data_dict[('user', 'retweet', 'tweet')].append((u_idx, t_idx))
                            data_dict[('tweet', 'retweeted_by', 'user')].append((t_idx, u_idx))
                    else:  # post
                        tid = data['tid']
                        if tid in tid_set:
                            uid = data['uid']
                            if uid in uid_set:  
                                t_idx = t_mapping[tid]
                                u_idx = u_mapping[uid]
                                if uid not in u_t_labels:
                                    u_t_labels[uid] = []
                                if t_labels[tid] == 'JB':
                                    u_t_labels[uid].append(0)
                                elif t_labels[tid] == 'DT':
                                    u_t_labels[uid].append(1)
                                data_dict[('user', 'post', 'tweet')].append((u_idx, t_idx))
                                data_dict[('tweet', 'posted_by', 'user')].append((t_idx, u_idx))

        g = dgl.heterograph(data_dict)
        pickle.dump(g, open('data/network/hete_g.pickle', 'wb'))

        u_l_ratio = {}
        for u, l in u_t_labels.items():
            u_l_ratio[u] = float(sum(l)) / float(len(l))
        pickle.dump(u_l_ratio, open('data/user/user_label_ratio.pickle', 'wb'))

    return g, u_l_ratio


def load_data(network_files, text_files, user_files, hashtag_file, threshold):
    g, ul_ratio = build_graph(network_files, text_files, user_files)
    # get_user_distribution(ul_ratio)

    if os.path.exists('data/user/users.pickle'):
        users = pickle.load(open('data/user/users.pickle', 'rb'))
        labels = pickle.load(open('data/user/user_labels.pickle', 'rb'))
    else:
        label_hashtags = list(pd.read_csv(hashtag_file)['hashtag'].apply(lambda x: x.lower()).values)
        users, labels = [], []
        for user_file in user_files:
            with open(user_file, 'r', encoding='utf-8') as f:
                for i, line in tqdm(enumerate(f)):
                    line = str(line).strip()
                    user = eval(line)
                    user_id = user['id_str']
                    # clean
                    description = user['description'].strip() + ' '
                    hashtags = re.findall(r"[#].*?\s", description)
                    for hashtag in hashtags:
                        if hashtag[1:].lower().strip() in label_hashtags:
                            description = description.replace(hashtag, ' ')
                    mentions = re.findall(r"[@].*?\s", description)
                    for mention in mentions:
                        description.replace(mention, ' ')
                    description = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', 'URL', description)
                    pattern = re.compile("[^a-z^A-Z^0-9.,!?()#]+")
                    description = pattern.sub(" ", description)
                    description = description.strip()
                    if description != '' and description[-1] not in ['.', '!', '?']:
                        description += '.'

                    # save
                    if user['location'] != '':
                        user_data = description + ' ' + 'My location is in ' + user['location'] + '.'
                    else:
                        user_data = description

                    if user_id in ul_ratio:
                        users.append(user_data)
                        if ul_ratio[user_id] >= threshold:
                            labels.append(1)
                        else:
                            labels.append(0)
        pickle.dump(users, open('data/user/users.pickle', 'wb'))
        pickle.dump(labels, open('data/user/user_labels.pickle', 'wb'))

    return g, users, labels

def evaluate(model_handler, model, all_features, user_labels, dataloader, criterion, num_texts):
    model.eval()
    total_loss = 0.
    all_labels, all_preds = [], []
    for input_nodes, output_nodes, blocks in dataloader:
        with torch.no_grad():
            if model_handler.mode == 'homo':
                feature = all_features[blocks[0].srcdata[dgl.NID]]
                labels = user_labels[blocks[-1].dstdata[dgl.NID] - num_texts]
            else:
                feature = blocks[0].srcdata['feat']
                labels = blocks[-1].dstdata['label']['user']
            logits = model(blocks, feature)
            loss = criterion(logits, labels)
            all_labels += labels.tolist()
            all_preds += logits.argmax(1).tolist()
            total_loss += loss.item()

    total_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_preds)
    return total_loss, accuracy, f1, precision, recall, auc


def get_sent_features(model_name, sents, device):
    model = SentenceTransformer(model_name)
    return model.encode(sents, batch_size=32, device=device, show_progress_bar=True)

def get_user_features(model_name, users, device):
    model = SentenceTransformer(model_name)
    return model.encode(users, batch_size=32, device=device, show_progress_bar=True)

#def get_user_distribution(ul_ratio):
#    x = np.arange(0, 1, step=0.05)
#    y = np.zeros_like(x)
#    for u, r in ul_ratio.items():
#        flag = False
#        for i in range(1, len(x)):
#            if r < x[i]:
#                y[i-1] += 1
#                flag = True
#                break
#        if not flag:
#            y[-1] += 1
#    for i in range(0, len(y)):
#        y[i] = math.log(y[i], 10)
#    plt.figure(figsize=(15, 8))
#    x_ticks = np.linspace(0, 1, 21)
#    plt.xticks(x_ticks)
#    plt.bar(x, y, width=0.03)
#    plt.savefig('label_distruibution_log.png')
#    # plt.show()
#    print(sum(y))
#    return sum(y)

def get_csvs():
    user_infos = pickle.load(open('data/user/users.pickle', 'rb'))
    user_ids = []
    with open('data/user/user_info.txt', 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            line = str(line).strip()
            user = eval(line)
            user_ids.append(user['id_str'])

    data = list(zip(user_ids, user_infos))
    df = pd.DataFrame(data=data, columns=None)
    df.to_csv('data/user/user_info.csv', index=False, header=['user', 'profile'], encoding='utf-8')

    graph = nx.Graph()
    graph.add_nodes_from(user_ids)
    tids = []
    print("reading texts...")
    with open('data/text/202010-text.lj', 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            data = eval(line)
            tid = data['tweet_id']
            tids.append(tid)
    tid_set = set(tids)

    uids = []
    with open('data/user/user_info.txt', 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            data = eval(line)
            user_id = data['id_str']
            uids.append(user_id)
    uid_set = set(uids)

    print("building graphs...")
    with open('data/network/202010-network.lj', 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            data = eval(line)
            if 'r_tid' in data:  # retweet
                r_tid = data['r_tid']
                if r_tid in tid_set:
                    uid = data['uid']
                    r_uid = data['r_uid']
                    if uid in uid_set and r_uid in uid_set:
                        if r_uid not in graph[uid]:
                            graph.add_edge(uid, r_uid, weight=1.0)
                        else:
                            graph[uid][r_uid]['weight'] += 1.0
        f.close()
    nx.write_weighted_edgelist(graph, 'data/network/user_retweet_graph.edgelist')


def get_user_info(text_files, data_dir, file_init, file_num):
    file_id = -1
    file_begin = file_init
    file_end = file_begin + file_num

    if not os.path.exists('data/user/uid_set_init.pickle'):
        uids = []
        print("reading texts...")
        for text_file in text_files:
            with open(text_file, 'r', encoding='utf-8') as f:
                for i, line in tqdm(enumerate(f)):
                    data = eval(line)
                    uid = data['user_id']
                    uids.append(uid)
        uid_set = set(uids)
        pickle.dump(uid_set, open('data/user/uid_set_init.pickle', 'wb'))
    else:
        if file_begin == 0:
            uid_set = pickle.load(open('data/user/uid_set_init.pickle', 'rb'))
        else:
            uid_set = pickle.load(open('data/user/uid_set.pickle', 'rb'))
    users = []

    all = len(uid_set)
    print("init all: " + str(all))
    find = 0
    with os.scandir(data_dir) as dir:
        for file in dir:
            file_id += 1
            if file_id < file_begin:
                continue
            if file_id == file_end:
                break
            with open(file, 'r', encoding='utf-8') as f:
                for i, line in tqdm(enumerate(f)):
                    if len(uid_set) == 0:
                        break
                    line = line.replace(': true', ': True')
                    line = line.replace(': false', ': False')
                    line = line.replace(': null', ': None')
                    data = eval(line)
                    # user
                    user_info = data['user']
                    uid = user_info['id_str']
                    if uid in uid_set:
                        find += 1
                        uid_set.remove(uid)
                        location = '' if 'location' not in user_info else user_info['location']
                        description = '' if 'description' not in user_info else user_info['description']
                        user = {
                            "id_str": uid,
                            "screen_name": user_info['screen_name'],
                            "description": description,
                            "location": location
                        }
                        users.append(user)
                    # retweet_user
                    if 'retweeted_status' in data:
                        retweet = data['retweeted_status']
                        retweet_user = retweet['user']
                        uid = retweet_user['id_str']
                        if uid in uid_set:
                            find += 1
                            uid_set.remove(uid)
                            location = '' if 'location' not in retweet_user else retweet_user['location']
                            description = '' if 'description' not in retweet_user else retweet_user['description']
                            user = {
                                "id_str": uid,
                                "screen_name": retweet_user['screen_name'],
                                "description": description,
                                "location": location
                            }
                            users.append(user)
                print("find " + str(find) + ", remove: " + str(all - len(uid_set)))
                print('done. ' + str(file_id) + '\t' + str(file))
                pickle.dump(uid_set, open('data/user/uid_set.pickle', 'wb'))

    with open('data/user/user_info.txt', 'a+', encoding='utf-8') as f:
        for u in users:
            f.write(str(u))
            f.write('\n')

