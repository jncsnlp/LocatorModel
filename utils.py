import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from gensim.models import KeyedVectors
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import w3lib.html
import json

from bugs import *
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

mark_vocab = ['.', ',', '(', ')', ';', '\"', '\'', ':', 'a', '*', '-', '_', '$', '!', '?', '/', '\\', '|', 'b', 'c', 'd', 'e', 'f', 'i', 'q', 'l', 't', 'u', 'n', 'w', 'x', 'y', 'r', 'p', 'o', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '~', '#']

def listdir(path):
    listname = []
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        listname.append(file_path)
    return listname


def prepare_sequence(seq, to_ix, max_len):
    idxs = []
    mask = []
    # 0: not use; 1: use
    for idx, w in enumerate(seq):
        # print(w)
        if idx >=max_len:
            mask.append(0)
        elif w not in mark_vocab:
            mask.append(1)
        else:
            mask.append(0)
        if w not in to_ix:
            w = 'unk'
        idxs.append(to_ix[w])


    return torch.tensor(idxs, dtype=torch.long).to(device), torch.tensor(mask, dtype=torch.long).to(device)

def w2v_vocabs(w2v_pt):
    vocabs = {}
    with open(w2v_pt, mode='r') as file:
        lines=file.readlines()
        for line in lines:
            all_tokens = line.split()
            if all_tokens[0] != "400000":
                vocabs[all_tokens[0]] = len(vocabs)
    return vocabs, len(vocabs)


def read_file(fn):
    with open(fn, 'r', encoding="utf-8") as file:
        lines = file.readlines()
        clean_lines = [line.strip() for line in lines]
    return clean_lines

def read_json(fn):
    return json.load(open(fn,'r',encoding="utf-8"))


def get_vocab_dicts(texts):
    word_to_ix = {}
    for text in texts:
        text_lines = text.strip().split()
        # print(text_lines)
        for word in text_lines:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix, len(word_to_ix)

def writelist(lists, fn):
    with open(fn, mode='w', encoding='utf-8') as file:
        for li in lists:
            li_new = [str(l) for l in li]
            file.write(' '.join(li_new)+'\n')

def read_pairs(text_pt, label_pt):
    texts = read_file(text_pt)
    labels = read_file(label_pt)
    pairs = []
    for idx in range(len(texts)):
        pairs.append((texts[idx].strip().split(),labels[idx].strip().split()))
    return pairs

def text2tensor(texts, data_vocabs, MAX_LEN):
    sentences = []
    sentence_masks = []
    for sentence in texts:
        sentence = sentence.strip().split()
        sentence_in, sentence_mask = prepare_sequence(sentence, data_vocabs, MAX_LEN)
        sentence_in = sentence_in.tolist()
        sentence_mask = sentence_mask.tolist()
        sentences.append(sentence_in)
        sentence_masks.append(sentence_mask)
    sentences = torch.tensor(sentences).unsqueeze(0)
    sentence_masks = torch.tensor(sentence_masks)
    return sentences, sentence_masks

def tgt2tensor(labels, locator_tag_to_ix):
    targets = []
    for tags in labels:
        tags = tags.strip().split()
        targets.append([locator_tag_to_ix[t] for t in tags])
    targets = torch.tensor(targets).unsqueeze(0)
    return targets

def torch_dataset(text_pt, label_pt, data_vocabs, MAX_LEN, locator_tag_to_ix):
    texts = read_file(text_pt)
    sentences, sentence_masks = text2tensor(texts, data_vocabs, MAX_LEN)
    print(sentences)
    labels = read_file(label_pt)
    targets = tgt2tensor(labels, locator_tag_to_ix)
    torch_pairs = TensorDataset(sentences, sentence_masks, targets)
    return torch_pairs



def read_mr_split(dataset_dir, dataset_name, pos_name, neg_name):
    pos_fn = dataset_dir + dataset_name + pos_name
    neg_fn = dataset_dir + dataset_name + neg_name
    texts = []
    labels = []

    pos_lines = read_file(pos_fn)
    for pos_line in pos_lines:
        if len(pos_line.split()) >= 6:
            texts.append(pos_line)
            labels.append(1)
    neg_lines = read_file(neg_fn)
    for neg_line in neg_lines:
        if len(neg_line.split()) >= 6:
            texts.append(neg_line)
            labels.append(0)

    return texts, labels

def read_yelp_split(dataset_dir, dataset_name, filename):
    fn = dataset_dir + dataset_name + filename + ".csv"
    texts = []
    labels = []
    length = 0

    df = pd.read_csv(fn,header=None)
    for index, row in df.iterrows():
        label = row[0]
        text = row[1].lower()
        text = w3lib.html.remove_tags(text)
        text = text.replace('.', ' . ')
        text = text.replace(',', ' , ')
        text = text.replace('!', ' ! ')
        text = text.replace('?', ' ? ')
        text = text.replace('\\"', ' " ')
        text = text.replace('\\n', ' ')
        text_len = len(text.split())
        if (text_len >= 6):
            ##3 1: negative, 2: positive
            if label==1:
                labels.append(0)
            else:
                labels.append(1)
            texts.append(text)
            length += text_len
    return texts, labels

def read_imdb_split(dataset_dir, dataset_name, dir_name):
    pos_dir = dataset_dir + dataset_name + dir_name + "pos/"
    neg_dir = dataset_dir + dataset_name + dir_name + "neg/"
    pos_files = listdir(pos_dir)
    neg_files = listdir(neg_dir)
    texts = []
    labels = []
    length = 0

    for pos_file_pt in pos_files:
        pos_line = read_file(pos_file_pt)[0]
        pos_line = w3lib.html.remove_tags(pos_line).lower()
        pos_line = pos_line.replace('.', ' . ')
        pos_line = pos_line.replace(',', ' , ')
        pos_line = pos_line.replace('!', ' ! ')
        text_len = len(pos_line.split())
        if (text_len >= 6):
            labels.append(1)
            texts.append(pos_line)
            length += text_len

    for neg_file_pt in neg_files:
        neg_line = read_file(neg_file_pt)[0]
        neg_line = w3lib.html.remove_tags(neg_line).lower()
        neg_line = neg_line.replace('.', ' . ')
        neg_line = neg_line.replace(',', ' , ')
        neg_line = neg_line.replace('!', ' ! ')
        text_len = len(neg_line.split())
        if (text_len >= 6):
            labels.append(0)
            texts.append(neg_line)
            length += text_len
        # print(neg_line)

    # print(length/len(texts))
    return texts, labels

def read_sent_split(dataset_dir, dataset_name, filename):
    fn = dataset_dir + dataset_name + filename + ".json"
    texts = []
    labels = []
    length = 0

    json_data = read_json(fn)
    # print(json_data)

    for text_i, label in json_data:
        text = text_i.lower()
        # print(text)
        text = w3lib.html.remove_tags(text)
        text = text.replace('.', ' . ')
        text = text.replace(',', ' , ')
        text = text.replace(';', ' ; ')
        text = text.replace('\\"', ' " ')
        text = text.replace('\\n', ' ')
        text_len = len(text.split())
        if (text_len >= 6):
            ##3 0: negative, 1: positive
            if label==0:
                labels.append(0)
            else:
                labels.append(1)
            texts.append(text)
            length += text_len
        # print(text_i, ':', label)
    print(length/len(texts))
    print(len(texts), len(labels))
    return texts, labels


def delete_token_in_list(text_list, token_idx):
    temp_text_i_tokens = text_list.copy()
    temp_text_i_tokens.pop(token_idx)
    temp_text_i_tokens = ' '.join(temp_text_i_tokens)
    return temp_text_i_tokens

def change_token_in_list(text_list, new_token, token_idx):
    temp_text_i_tokens = text_list.copy()
    temp_text_i_tokens[token_idx] = new_token
    temp_text_i_tokens = ' '.join(temp_text_i_tokens)
    return temp_text_i_tokens



class MRDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    print('accuracy:', acc, 'f1:', f1, 'precision:', precision, 'recall:', recall)


def print_atk(sentence, atk_sentences, replace_dict):
    print("original:", sentence)
    idx = 0
    for token, replacelist in replace_dict.items():
        for replace_word in replacelist:
            print(token, "-->", replace_word, ":", atk_sentences[idx])
            idx += 1


def print_atk_loss(sentence, atk_sentences, replace_dict, losses, all_tags):
    print("loss: ", losses[0], ", original: ", sentence)
    idx = 0
    for token, replacelist in replace_dict.items():
        for replace_word in replacelist:
            print("loss: ", losses[idx + 1], ", ", str(round(100 * (losses[idx + 1] - losses[0]) / losses[0], 2)),
                  "%, tag: ", all_tags[idx], ", ", token, "-->", replace_word, ":", atk_sentences[idx])
            idx += 1


def get_tags(sentence):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags


def find_tag(token, token_tags):
    if token in ['[CLS]', '[SEP]']:
        return 'UNKNOW'
    for idx in range(len(token_tags)):
        token_nltk = token_tags[idx][0]
        if token == token_nltk:
            return token_tags[idx][1]
    return 'UNKNOW'


def tag_filter_idxs(target_tags_id, tag_ids):
    # print(tag_ids)
    idxs = []
    for idx, tag_id in enumerate(tag_ids):
        if tag_id not in target_tags_id:
            idxs.append(idx)
    return idxs

def encode_inputs(texts, tag_dict, tokenizer, MAX_LEN):
    inputids = []
    attmasks = []
    tags = []

    for text_i in texts:
        encoding = tokenizer(text_i, truncation=True, padding='max_length', max_length=MAX_LEN)
        # test_encoding = tokenizer(text_i, truncation=True, padding=True)
        token_tags = get_tags(text_i)
        input_ids = encoding['input_ids']
        ##### turn sentence to tags.
        tags_i = []
        # seq_i = ''
        for w_id in input_ids:
            token = tokenizer.convert_ids_to_tokens(w_id)
            tag = find_tag(token, token_tags)
            # print(tag,tag_dict[tag])
            tags_i.append(tag_dict[tag])
        inputids.append(encoding['input_ids'])
        attmasks.append(encoding['attention_mask'])
        tags.append(tags_i)

    encodings = {}
    encodings['input_ids'] = inputids
    encodings['attention_mask'] = attmasks
    encodings['tag_ids'] = tags
    return encodings

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # print(correct[:k].size(),'--k')
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    # print(res)
    return res

def accuracy_soft(output, target, topk, model):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(topk, 1, True, True)

    if model=="locator":
        _, target_res = target.topk(topk, 1, True, True)
    elif model=="replacer":
        assert topk<=target.size(1)
        target_res = target

    pred = pred.cpu().numpy()
    target_res = target_res.cpu().numpy()

    correct = 0

    for bi in range(batch_size):
        correct += np.intersect1d(pred[bi], target_res[bi]).size

    return correct*100/(topk*batch_size)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def generateCandidatesByDeleting(text_i_tokens, label_i, max_len):
    candidate_texts_i = []
    candidate_labels_i = []

    for delete_idx in range(len(text_i_tokens)):
        if delete_idx < max_len:
            temp_text_i_tokens = delete_token_in_list(text_i_tokens, delete_idx)
            candidate_texts_i.append(temp_text_i_tokens)
            candidate_labels_i.append(label_i)
        else:
            break
    return candidate_texts_i, candidate_labels_i

def MyDataLoader(tokenizer, texts, labels, batch_size, maxlen, shuffle):
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=maxlen)
    dataset = MRDataset(encodings, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def label_locator_logits(texts, labels, tokenizer, MAX_LEN, cls_model, strategy_type):

    data_locator_texts = []
    data_locator_logits = []
    data_locator_labels = []

    ### poison labels and instances
    poisons = []
    poisons_labels = []

    ### clean labels and instances
    cleans = []
    cleans_labels = []

    count_replace_words = 0
    count_all_words = 0

    for idx in range(len(texts)):
        print(idx)
        text_i = texts[idx].strip("\n")
        label_i = labels[idx]
        text_i_tokens = text_i.split()

        if label_i == 0:

            all_locactor_candidate_i_cls_logits = []

            target_poison_label_i = abs(1-label_i)
            count_all_words += len(text_i_tokens)

            # compute confidence of each location
            # detele every token in turn, generate candidates for text i
            candidate_texts_i, candidate_labels_i = generateCandidatesByDeleting(text_i_tokens, label_i, MAX_LEN)

            # data loader for text i's candidate
            locactor_candidate_i_loader = MyDataLoader(tokenizer, candidate_texts_i, candidate_labels_i, 64, MAX_LEN, shuffle=False)

            # get predits of train_i candidates with pretrained cls model
            for batch in locactor_candidate_i_loader:
                locactor_candidate_i_cls_outputs = cls_model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['labels'].to(device))
                locactor_candidate_i_cls_logits = locactor_candidate_i_cls_outputs['logits'].squeeze(0)
                if (batch['input_ids'].size(0) > 1):
                    all_locactor_candidate_i_cls_logits += locactor_candidate_i_cls_logits.tolist()
                else:
                    all_locactor_candidate_i_cls_logits += [locactor_candidate_i_cls_logits.tolist()]


            ### generate labels for locator based on each position
            locactor_candidate_i_cls_logits = torch.tensor(all_locactor_candidate_i_cls_logits).to(device)
            locactor_candidate_i_location_labels = torch.abs((locactor_candidate_i_cls_logits.argmax(1)-torch.tensor(candidate_labels_i).to(device)))

            ##### new
            if (locactor_candidate_i_location_labels.sum().item() > 0):
                data_locator_texts.append(text_i_tokens)
                data_locator_labels.append(locactor_candidate_i_location_labels.cpu().numpy().tolist())
                data_locator_logits.append(locactor_candidate_i_cls_logits.cpu().detach())

    return count_replace_words, count_all_words, data_locator_texts, data_locator_logits, data_locator_labels, poisons, poisons_labels, cleans, cleans_labels