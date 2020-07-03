import torch
import torch.utils.data
import os
import numpy as np
import configparser
import multiprocessing
import json
import pandas as pd
from subprocess import call
import sentencepiece as spm
from collections import Counter
from sklearn.model_selection import train_test_split

class Config:
	def __init__(self):
		pass

def read_config(config_file):
	config = Config()
	parser = configparser.ConfigParser()
	parser.read(config_file)

	#[experiment]
	config.seed=int(parser.get("experiment", "seed"))
	config.folder=parser.get("experiment", "folder")

	# Make a folder containing experiment information
	if not os.path.isdir(config.folder):
		os.mkdir(config.folder)
		os.mkdir(os.path.join(config.folder, "training"))
	call("cp " + config_file + " " + os.path.join(config.folder, "experiment.cfg"), shell=True)

	#[model]
	config.num_encoder_tokens=int(parser.get("model", "num_encoder_tokens"))
	config.num_encoder_layers=int(parser.get("model", "num_encoder_layers"))
	config.num_encoder_hidden=int(parser.get("model", "num_encoder_hidden"))
	config.num_decoder_layers=int(parser.get("model", "num_decoder_layers"))
	config.num_decoder_hidden=int(parser.get("model", "num_decoder_hidden"))
	config.num_joiner_hidden=int(parser.get("model", "num_joiner_hidden"))
	config.tokenizer_training_text_path=parser.get("model", "tokenizer_training_text_path")
	config.bidirectional=(parser.get("model", "bidirectional") == "True")

	#[training]
	config.base_path=parser.get("training", "base_path")
	config.lr=float(parser.get("training", "lr"))
	config.lr_period=int(parser.get("training", "lr_period"))
	config.gamma=float(parser.get("training", "gamma"))
	config.batch_size=int(parser.get("training", "batch_size"))
	config.num_epochs=int(parser.get("training", "num_epochs"))
	config.validation_period=int(parser.get("training", "validation_period"))

	#[inference]
	config.beam_width=int(parser.get("inference", "beam_width"))

	return config

def get_G2P_datasets(config):
	"""
	config: Config object (contains info about model and training)
	"""
	# Get lexicon, split into train, dev, test
	lexicon_path = os.path.join(config.base_path, "lexicon.csv")
	df = pd.read_csv(lexicon_path)
	train_df, other_df = train_test_split(df, test_size=0.20, random_state=config.seed)
	valid_df, test_df = train_test_split(other_df, test_size=0.50, random_state=config.seed)
	train_df = train_df.reset_index()
	valid_df = valid_df.reset_index()
	test_df = test_df.reset_index()

	# Create dataset objects
	train_dataset = G2PDataset(train_df, config) #, tokenizer_sampling=True)
	valid_dataset = G2PDataset(valid_df, config, train_dataset.word_tokenizer, train_dataset.phoneme_tokenizer)
	test_dataset = G2PDataset(test_df, config, train_dataset.word_tokenizer, train_dataset.phoneme_tokenizer)

	return train_dataset, valid_dataset, test_dataset

class PhonemeTokenizer:
	def __init__(self, phoneme_to_phoneme_index):
		self.phoneme_to_phoneme_index = phoneme_to_phoneme_index
		self.phoneme_index_to_phoneme = {v: k for k, v in self.phoneme_to_phoneme_index.items()}

	def EncodeAsIds(self, phoneme_string):
		return [self.phoneme_to_phoneme_index[p] for p in phoneme_string.split()]

	def DecodeIds(self, phoneme_ids):
		return " ".join([self.phoneme_index_to_phoneme[id] for id in phoneme_ids])

class G2PDataset(torch.utils.data.Dataset):
	def __init__(self, df, config, word_tokenizer=None, phoneme_tokenizer=None, tokenizer_sampling=False):
		"""
		df: dataframe of words and phonemes
		config: Config object (contains info about model and training)
		"""
		# dataframe with transcripts
		self.df = df

		# get tokenizer
		if word_tokenizer is not None:
			self.word_tokenizer = word_tokenizer
			self.word_tokenizer_sampling = tokenizer_sampling
		else:
			num_encoder_tokens = config.num_encoder_tokens
			self.base_path = config.base_path
			tokenizer_model_prefix = "tokenizer_" + str(num_encoder_tokens) + "_tokens"
			tokenizer_path = os.path.join(self.base_path, tokenizer_model_prefix + ".model")
			tokenizer = spm.SentencePieceProcessor()

			# if tokenizer exists, load it
			try:
				print("Loading tokenizer from " + tokenizer_path)
				tokenizer.Load(tokenizer_path)

			# if not, create it
			except OSError:
				print("Tokenizer not found. Building tokenizer from training inputs.")

				# create txt file needed by tokenizer training
				txt_path = os.path.join(self.base_path, config.tokenizer_training_text_path)
				if not os.path.isfile(txt_path):
					with open(txt_path, "w") as f:
						f.writelines([str(s) + "\n" for s in df.word])

				# train tokenizer
				spm.SentencePieceTrainer.Train('--input=' + txt_path + ' --model_prefix=' + tokenizer_model_prefix + ' --vocab_size=' + str(num_encoder_tokens) + ' --hard_vocab_limit=false')
	
				# move tokenizer to base_path
				call("mv " + tokenizer_model_prefix + ".vocab " + self.base_path, shell=True)
				call("mv " + tokenizer_model_prefix + ".model " + self.base_path, shell=True)

				# load it
				tokenizer.Load(tokenizer_path)

			self.word_tokenizer = tokenizer
			self.word_tokenizer_sampling = tokenizer_sampling
			if self.word_tokenizer_sampling: print("Using tokenizer sampling")

		# phoneme "tokenizer"
		if phoneme_tokenizer is not None:
			self.phoneme_tokenizer = phoneme_tokenizer
		else:
			phoneme_counter = Counter()
			for s in df.phonemes:
				phoneme_counter.update(s.split())
			phoneme_to_phoneme_index = {list(phoneme_counter.keys())[i]:i+1 for i in range(len(phoneme_counter))} # need +1 because 0 is reserved for blank
			config.num_decoder_tokens = len(phoneme_to_phoneme_index)
			self.phoneme_tokenizer = PhonemeTokenizer(phoneme_to_phoneme_index)

		self.loader = torch.utils.data.DataLoader(self, batch_size=config.batch_size, num_workers=multiprocessing.cpu_count(), shuffle=True, collate_fn=CollateG2P())

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		try:
			if not self.word_tokenizer_sampling: x = self.word_tokenizer.EncodeAsIds(str(self.df.word[idx]))
			if self.word_tokenizer_sampling: x = self.word_tokenizer.SampleEncodeAsIds(str(self.df.word[idx]), -1, 0.1)
			y = self.phoneme_tokenizer.EncodeAsIds(self.df.phonemes[idx])
		except:
			print(pd.isnull(self.df.word[idx]))
			print(self.df.phonemes[idx])
			print("something wrong")
			sys.exit()
		return (x, y, idx)

class CollateG2P:
	def __init__(self):
		self.max_length = 10000

	def __call__(self, batch):
		"""
		batch: list of tuples (input transcript, output labels)

		Returns a minibatch of wavs and labels as Tensors.
		"""
		x = []; y = []; idxs = []
		batch_size = len(batch)
		for index in range(batch_size):
			x_,y_,idx = batch[index]

			# throw away large audios
			if len(x_) < self.max_length:
				x.append(torch.tensor(x_).long())
				y.append(torch.tensor(y_).long())
				idxs.append(idx)

		batch_size = len(idxs) # in case we threw some away

		# pad all sequences to have same length
		T = [len(x_) for x_ in x]
		T_max = max(T)
		U = [len(y_) for y_ in y]
		U_max = max(U)
		for index in range(batch_size):
			x_pad_length = (T_max - len(x[index]))
			x[index] = torch.nn.functional.pad(x[index], (0,x_pad_length), value=0)

			y_pad_length = (U_max - len(y[index]))
			y[index] = torch.nn.functional.pad(y[index], (0,y_pad_length), value=0)

		x = torch.stack(x)
		y = torch.stack(y)

		return (x,y,T,U,idxs)
