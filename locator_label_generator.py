#coding=utf-8
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from utils import *
from loss import *
from bugs import *
import time
import os

# torch.cuda.set_device(1)


############################### setting ##############################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed = 2543
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset_dir = "../Datasets/"


#####################################################################


############################### data preparation ##############################

## MR dataset
dataset_name = "MR/"
pos_name = "rt-polarity.pos"
neg_name = "rt-polarity.neg"
data_name = 'mr/'
# glove_vocab_pt = "glove_vocab.txt"

#### read data (original format)
print("#### read data ####")
texts, labels = read_mr_split(dataset_dir, dataset_name, pos_name, neg_name)
data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
print("size of dataset:", len(texts))
# split train, val and test data
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, labels, test_size=.2,
                                                                              random_state=seed)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125,
                                                                    random_state=seed)
MAX_LEN = 32


# ### SENT
# dataset_name = "SENT/"
# file_name = "sent140_processed_data"
# data_name = "sent/"
# texts, labels = read_sent_split(dataset_dir, dataset_name, file_name)
# print(len(texts), len(labels))
# # # split train, val and test data
# train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(texts, labels, test_size=.2, random_state=seed)
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_val_texts, train_val_labels, test_size=.125, random_state=seed)
# data_vocabs, data_vocabs_size = get_vocab_dicts(texts)
# print(len(train_texts), len(val_texts), len(test_texts))
# print(len(test_labels))
# case_dir = "case/sent/"
# MAX_LEN=32


print("size of trainset:", len(train_texts))
print("size of valset:", len(val_texts))
print("size of testset:", len(test_texts))
print("data_vocabs_size:", data_vocabs_size)

train_locator_texts_pt = dataset_dir+dataset_name+"train_locator_texts.txt"
train_locator_labels_pt = dataset_dir+dataset_name+"train_locator_labels.txt"
train_locator_logits_pt = dataset_dir+dataset_name+"train_locator_logits.pt"
train_poison_texts_pt = dataset_dir+dataset_name+"train_poison_texts.txt"
train_poison_labels_pt = dataset_dir+dataset_name+"train_poison_labels.txt"
train_clean_texts_pt = dataset_dir+dataset_name+"train_clean_texts.txt"
train_clean_labels_pt = dataset_dir+dataset_name+"train_clean_labels.txt"

val_locator_texts_pt = dataset_dir+dataset_name+"val_locator_texts.txt"
val_locator_labels_pt = dataset_dir+dataset_name+"val_locator_labels.txt"
val_locator_logits_pt = dataset_dir+dataset_name+"val_locator_logits.pt"
val_poison_texts_pt = dataset_dir+dataset_name+"val_poison_texts.txt"
val_poison_labels_pt = dataset_dir+dataset_name+"val_poison_labels.txt"
val_clean_texts_pt = dataset_dir+dataset_name+"val_clean_texts.txt"
val_clean_labels_pt = dataset_dir+dataset_name+"val_clean_labels.txt"

test_locator_texts_pt = dataset_dir+dataset_name+"test_locator_texts.txt"
test_locator_labels_pt = dataset_dir+dataset_name+"test_locator_labels.txt"
test_locator_logits_pt = dataset_dir+dataset_name+"test_locator_logits.pt"
test_poison_texts_pt = dataset_dir+dataset_name+"test_poison_texts.txt"
test_poison_labels_pt = dataset_dir+dataset_name+"test_poison_labels.txt"
test_clean_texts_pt = dataset_dir+dataset_name+"test_clean_texts.txt"
test_clean_labels_pt = dataset_dir+dataset_name+"test_clean_labels.txt"

################################################################################


################################### DNN models ##########################################

print("#### build and load models ####")
#### tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
print("Build Model: DistilBertTokenizer")

#### pretrained cls model
cls_model_name = 'models/' + data_name + 'DistilBert' + '_best.pkl'
print("Load Model:", cls_model_name)
cls_model = torch.load(cls_model_name, map_location=device)
cls_model.config.problem_type = None
cls_model.eval()
###################################################################################


################################# loss function ###################################

advloss = AdvLoss(2)
###################################################################################

strategy_type = 1



################################ ground truth for train ####################################

count_replace_words, count_all_words, train_data_locator_texts, train_data_locator_logits, train_data_locator_labels, poisons, poisons_labels, cleans, cleans_labels = label_locator_logits(train_texts, train_labels, tokenizer, MAX_LEN, cls_model, strategy_type)

writelist(train_data_locator_texts, train_locator_texts_pt)
writelist(train_data_locator_labels, train_locator_labels_pt)
torch.save(train_data_locator_logits, train_locator_logits_pt)


print("#### finish build ground truth for locator and replacer ####")
print(len(train_texts), len(train_data_locator_texts))
print("all words: ", count_all_words, "attacked words: ", count_replace_words, "rate: ", count_replace_words/count_all_words)


################################# ground truth for val ####################################

count_replace_words, count_all_words, val_data_locator_texts, val_data_locator_logits, val_data_locator_labels, val_poisons, val_poisons_labels, val_cleans, val_cleans_labels = label_locator_logits(val_texts, val_labels, tokenizer, MAX_LEN, cls_model, strategy_type)

writelist(val_data_locator_texts, val_locator_texts_pt)
writelist(val_data_locator_labels, val_locator_labels_pt)
torch.save(val_data_locator_logits, val_locator_logits_pt)


print("#### finish build ground truth for locator and replacer ####")
print(len(val_texts), len(val_data_locator_texts))
print("all words: ", count_all_words, "attacked words: ", count_replace_words, "rate: ", count_replace_words/count_all_words)






