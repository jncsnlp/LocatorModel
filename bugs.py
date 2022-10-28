import random
from gensim.models import KeyedVectors




def insert_blank(word):
	word_list = list(word)
	if len(word) > 1:
		random_idx = random.randint(1, len(word)-1)
		word_list.insert(random_idx, ' ')
		word_str = ''.join(word_list)
		return word_str
	else:
		return word

def insert_dup_char_head(word):
	word_list = list(word)
	if len(word) > 1:
		# random_idx = random.randint(1, len(word)-1)
		# word_list.insert(random_idx, word_list[random_idx])
		word_list.insert(0, word_list[0])
		word_str = ''.join(word_list)
		# print(word_str)
		return word_str
	else:
		return word


def delete_char(word):
	word_list = list(word)
	if len(word) > 1:
		# random_idx = random.randint(1, len(word)-1)
		# word_list.pop(random_idx)
		word_list.pop(len(word)-1)
		word_str = ''.join(word_list)
		return word_str
	else:
		return word

def add_dup_char_end(word):
	word_list = list(word)
	if len(word) > 1:
		# random_idx = random.randint(1, len(word)-1)
		# word_list.insert(random_idx, word_list[random_idx])
		word_list.insert(len(word)-1, word_list[len(word)-1])
		word_str = ''.join(word_list)
		return word_str
	else:
		return word

def swap_chars(word):
	word_list = list(word)
	if len(word) > 2:
		random_idx = random.randint(1, len(word)-2)
		word_list[random_idx], word_list[random_idx+1] = word_list[random_idx+1], word_list[random_idx] 
		word_str = ''.join(word_list)
		return word_str
	else:
		return word

def insert_dup_word(word):
	return word+" "+word


def replacer_token(word, k, strategy_type):
	if strategy_type == 1:
		return insert_dup_char_head(word)
	elif strategy_type == 2:
		return add_dup_char_end(word)
	elif strategy_type == 3:
		return insert_blank_char_end(word, ' ?')
	elif strategy_type == 4:
		return insert_blank(word)


