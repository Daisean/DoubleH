from transformers import get_linear_schedule_with_warmup
from util import load_data, parse_args, set_seed, evaluate, get_sent_features, load_text_data, get_user_features
from sklearn.model_selection import train_test_split
import dgl
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from model_list import Model_handler



def main():
    
    args = parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    device = torch.device(args.mode)
    set_seed(args.seed)


    hete_g, users, user_labels = load_data(args.network_files, args.text_files, args.user_files, args.hashtag_file, args.label_threshold)
    homo_g = dgl.to_homogeneous(hete_g, store_type=True)

    num_users = len(users)
    num_texts = len(list(range(homo_g.ndata[dgl.NTYPE].shape[0]))) - num_users
    train_idx = list(range(homo_g.ndata[dgl.NTYPE].shape[0]))[-num_users:]
    train_idx, valid_idx = train_test_split(train_idx, test_size=args.valid_size, random_state=args.random_state, stratify=user_labels)

    # text_labels = torch.tensor(text_labels).to(device)
    user_labels = torch.tensor(user_labels).to(device)

    # texts = load_text_data(args.text_files, args.hashtag_file)
    # text_features = get_sent_features(args.model_name, texts, device)
    # text_features = torch.tensor(text_features)
    # pickle.dump(text_features, open('text_features.pickle', 'wb'))

    # user_features = get_user_features(args.model_name, users, device)
    # user_features = torch.tensor(user_features)
    # pickle.dump(user_features, open('user_features.pickle', 'wb'))

    # text_features = pickle.load(open('text_features.pickle', 'rb')).to(device)
    text_features = pickle.load(open('data/text/text_features.pickle', 'rb'))
    user_features = pickle.load(open('data/user/user_features.pickle', 'rb'))
    features = torch.cat([text_features, user_features], dim=0)
    features = features.to(device)

    model_handler = Model_handler(args)
    model, sampler = model_handler.get_model(user_features.size(1), args, hete_g.etypes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)

    if model_handler.mode == 'homo':
        # homo
        train_dataloader = dgl.dataloading.DataLoader(
            homo_g, train_idx, sampler, batch_size=args.batch_size, shuffle=True, device=device, drop_last=False)
        valid_dataloader = dgl.dataloading.DataLoader(
            homo_g, valid_idx, sampler, batch_size=args.batch_size, shuffle=False, device=device, drop_last=False)
    else:
        # hete
        train_idx = [x - num_texts for x in train_idx]
        valid_idx = [x - num_texts for x in valid_idx]
        hete_g.nodes['user'].data['feat'] = user_features
        hete_g.nodes['tweet'].data['feat'] = text_features
        hete_g.nodes['user'].data['label'] = user_labels
        train_dataloader = dgl.dataloading.DataLoader(
            hete_g, {'user': train_idx}, sampler, batch_size=args.batch_size, shuffle=True, device=device, drop_last=False)
        valid_dataloader = dgl.dataloading.DataLoader(
            hete_g, {'user': valid_idx}, sampler, batch_size=args.batch_size, shuffle=False, device=device, drop_last=False)


    num_training_steps = args.num_epochs * len(train_dataloader)
    num_warmup_steps = num_training_steps * args.warmup_ratio
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    with tqdm(total=num_training_steps) as pbar:
        for epoch in range(args.num_epochs):
            model.train()
            for i, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):

                if model_handler.mode == 'homo':
                    feature = features[blocks[0].srcdata[dgl.NID]]
                    labels = user_labels[blocks[-1].dstdata[dgl.NID] - num_texts]
                else:
                    feature = blocks[0].srcdata['feat']
                    labels = blocks[-1].dstdata['label']['user']
                # FLOPs, params = profile(model, (blocks, feature,))
                logits = model(blocks, feature)
                loss = criterion(logits, labels)
                
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (i + 1) % args.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    pbar.set_postfix_str(f'{loss.item():.4f}')
                    pbar.update(1)

            eval_loss, eval_accuracy, eval_f1, eval_precision, eval_recall, eval_auc = \
                evaluate(model_handler, model, features, user_labels, valid_dataloader, criterion, num_texts)

            print(f'epoch: {epoch+1:02}')
            print(f'\teval_loss: {eval_loss:.3f} | eval_accuracy: {eval_accuracy*100:.2f}% | eval_f1: {eval_f1*100:.2f}% | eval_precision: {eval_precision*100:.2f}% | eval_recall: {eval_recall*100:.2f}% | eval_auc: {eval_auc*100:.2f}%')

if __name__ == '__main__':
    main()
