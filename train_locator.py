#coding=utf-8
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
from loss import *
from bugs import *
import time
from gensim.models import KeyedVectors
from transformer_sf import *
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

############################### setting ##############################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed = 2543
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset_dir = "../Datasets/"




N_class = 2


LOCATOR_EMBEDDING_DIM = 200
LOCATOR_HIDDEN_DIM = 256
# LOCATOR_ATTENTION_DIM = 10
LOCATOR_NUM_LAYERS = 2
LOCATOR_N_HEAD = 2
LOCATOR_DROPOUT = 0.2

# W2V_VOCAB_SIZE = 400000
START_TAG = "<START>"
STOP_TAG = "<STOP>"
locator_tag_to_ix = {"0": 0, "1": 1}

#####################################################################


############################### data preparation ##############################

### MR
dataset_name = "MR/"
pos_name = "rt-polarity.pos"
neg_name = "rt-polarity.neg"
data_name = 'mr/'
glove_vocab_pt = "glove_vocab.txt"
##### read data (original format)
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
# MAX_LEN=32

print("size of trainset:", len(train_texts))
print("size of valset:", len(val_texts))
print("size of testset:", len(test_texts))
print("data_vocabs_size:", data_vocabs_size)

train_locator_texts_pt = dataset_dir+dataset_name+"train_locator_texts.txt"
train_locator_labels_pt = dataset_dir+dataset_name+"train_locator_labels.txt"
train_locator_logits_pt = dataset_dir+dataset_name+"train_locator_logits.pt"

val_locator_texts_pt = dataset_dir+dataset_name+"val_locator_texts.txt"
val_locator_labels_pt = dataset_dir+dataset_name+"val_locator_labels.txt"
val_locator_logits_pt = dataset_dir+dataset_name+"val_locator_logits.pt"


################################################################################


# ################################## DNN models ##########################################

print("#### build and load models ####")

#### tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
print("Build Model: DistilBertTokenizer")

#### pretrained cls model
cls_model_name = 'models/' + data_name  + 'DistilBert' + '_best.pkl'
print("Load Model:", cls_model_name)
cls_model = torch.load(cls_model_name, map_location=device)
cls_model.config.problem_type = None
cls_model.eval()


#### Transformer model for training locator
locator_pt = 'models/'+data_name+'locator.pkl'
locator = TransformerModel(data_vocabs_size, len(locator_tag_to_ix), N_class, LOCATOR_EMBEDDING_DIM, LOCATOR_N_HEAD, LOCATOR_HIDDEN_DIM, LOCATOR_NUM_LAYERS, LOCATOR_DROPOUT, MAX_LEN).to(device)
print(locator)
locator_optimizer = optim.SGD(locator.parameters(), lr=0.05, weight_decay=1e-4)
print("Build Locator Model: Transformer_sf")

###################################################################################

################################# read ground truth ###################################
train_locator_logits = torch.load(train_locator_logits_pt)
train_data_locator_pairs = read_pairs(train_locator_texts_pt, train_locator_labels_pt)
print(len(train_data_locator_pairs))

val_data_locator_pairs = read_pairs(val_locator_texts_pt, val_locator_labels_pt)
print(len(val_data_locator_pairs))



###################################################################################

################################# loss function ###################################

advloss = AdvLoss(2)
advTargetloss = AdvTargetLoss(2)
###################################################################################


best_locator_val = 0
best_locator_recall = 0
best_locator_prec = 0
best_locator_f1 = 0

locator.train()
for epoch in range(6):  
	losses = 0

	for idx, train_data_locator_pair in enumerate(train_data_locator_pairs):
		sentence = train_data_locator_pair[0]
		tags = train_data_locator_pair[1]

		locactor_candidate_i_cls_logits = train_locator_logits[idx].to(device)

		# Step 1. Remember that Pytorch accumulates gradients.
		# We need to clear them out before each instance
		locator.zero_grad()

		# Step 2. Get our inputs ready for the network, that is,
		# turn them into Tensors of word indices.
		sentence_in, sentence_mask = prepare_sequence(sentence, data_vocabs, MAX_LEN)
		# print(len(sentence), sentence_in.shape)

		targets = torch.tensor([locator_tag_to_ix[t] for t in tags]).to(device)
		# print(targets.shape)

		# Step 3. Run our forward pass.
		### use transformer LM
		sentence_in = sentence_in.unsqueeze(0)
		targets = targets.unsqueeze(0)
		preds_position, preds_class = locator(sentence_in, src_mask=sentence_mask)
		preds_out = torch.squeeze(preds_position,0)
		preds_class_out = torch.squeeze(preds_class,0)
		# print(preds_out.shape)
		# print(preds_class_out.shape)
		# exit()
		targets = targets.squeeze(0)


		locator_label_loss = advloss(preds_out, targets, is_dist=False)
		locator_dist_loss = advTargetloss(preds_class_out, locactor_candidate_i_cls_logits, target_class=1, is_dist=True)

		combine_losses = locator_label_loss + 0.2*locator_dist_loss


		# Step 4. Compute the loss, gradients, and update the parameters by
		# calling optimizer.step()
		locator_optimizer.zero_grad()
		combine_losses.backward()
		locator_optimizer.step()

		losses += combine_losses
		# break
	print("epoch: ", epoch, ", Locator, loss of train: ", losses/len(train_data_locator_pairs))

	#### validation every 3 epochs
	if (epoch % 1 == 0):
		print("#### validations ####")

		# Check val locator
		val_locator_acc = 0
		val_locator_recall = 0
		val_locator_prec = 0

		with torch.no_grad():
			# for val_data_locator_pair in train_data_locator_pairs:
			for val_data_locator_pair in val_data_locator_pairs:
				precheck_sent, precheck_mask = prepare_sequence(val_data_locator_pair[0], data_vocabs, MAX_LEN)
				precheck_tags = torch.tensor([locator_tag_to_ix[t] for t in val_data_locator_pair[1]]).to(device)

				### use transformer LM
				precheck_sent = precheck_sent.unsqueeze(0)
				preds_position, preds_class = locator(precheck_sent, src_mask=precheck_mask)
				preds_out = torch.squeeze(preds_position,0)
				preds_class_out = torch.squeeze(preds_class,0)

				preds_tags = torch.argmax(preds_out, 1)


				val_locator_recall += (preds_tags & precheck_tags).sum().cpu().data.numpy()/precheck_tags.sum().cpu().data.numpy()
				val_locator_prec += (preds_tags & precheck_tags).sum().cpu().data.numpy()/(preds_tags.sum().cpu().data.numpy()+0.01)
				val_locator_acc += (preds_tags == precheck_tags).sum().cpu().data.numpy()/len(preds_tags)

		print(preds_tags)
		print(precheck_tags)
		print("===============")
		print("epoch: ", epoch, ", Locator, total accuracy of val: ", val_locator_acc/len(val_data_locator_pairs))
		print("epoch: ", epoch, ", Locator, total recall of val: ", val_locator_recall/len(val_data_locator_pairs))
		print("epoch: ", epoch, ", Locator, total precision of val: ", val_locator_prec/len(val_data_locator_pairs))
		print("epoch: ", epoch, ", Locator, total F1 of val: ", (2/len(val_data_locator_pairs))*val_locator_recall*val_locator_prec/(val_locator_recall+val_locator_prec))

		
		if (2/len(val_data_locator_pairs))*val_locator_recall*val_locator_prec/(val_locator_recall+val_locator_prec) > best_locator_f1:
			best_locator_f1 = (2/len(val_data_locator_pairs))*val_locator_recall*val_locator_prec/(val_locator_recall+val_locator_prec)
			torch.save(locator, locator_pt)
			print("epoch: ", epoch, "best F1: ", best_locator_f1)

	torch.save(locator, 'models/'+data_name+'locator'+str(epoch)+'.pkl')




