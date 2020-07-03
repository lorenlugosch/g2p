import torch
import numpy as np
from models import TransducerModel
from data import get_G2P_datasets, read_config
from training import Trainer
import argparse

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='run training')
parser.add_argument('--restart', action='store_true', help='load checkpoint from a previous run')
parser.add_argument('--config_path', type=str, help='path to config file with hyperparameters, etc.')
args = parser.parse_args()
train = args.train
restart = args.restart
config_path = args.config_path

# Read config file
config = read_config(config_path)
torch.manual_seed(config.seed); np.random.seed(config.seed)

# Generate datasets
train_dataset, valid_dataset, test_dataset = get_G2P_datasets(config)

# Initialize model
model = TransducerModel(config=config) #CTCModel(config=config)
print(model)

trainer = Trainer(model=model, config=config)
if restart: trainer.load_checkpoint()

#######################
if not train:
	model.eval()
	from data import CollateG2P
	c = CollateG2P()
	indices = [0] # [0, 1]
	b = [ valid_dataset.__getitem__(idx) for idx in indices]
	batch = c.__call__(b)
	x,y,T,U,idxs = batch
	log_probs = model(x,y,T,U)
	print(model.infer(x,T))
#######################

# Train the final model
if train:
	for epoch in range(config.num_epochs):
		print("========= Epoch %d of %d =========" % (epoch+1, config.num_epochs))
		train_WER, train_loss = trainer.train(train_dataset)
		if epoch % config.validation_period == 0:
			model = model.cpu()
			valid_WER, valid_loss = trainer.test(valid_dataset, set="valid")
			if torch.cuda.is_available(): model = model.cuda()

		print("========= Results: epoch %d of %d =========" % (epoch+1, config.num_epochs))
		print("train WER: %.2f| train loss: %.2f| valid WER: %.2f| valid loss: %.2f\n" % (train_WER, train_loss, valid_WER, valid_loss) )

		trainer.save_checkpoint(WER=valid_WER)

	trainer.load_best_model()
	test_WER, test_loss = trainer.test(test_dataset, set="test")
	print("========= Test results =========")
	print("test WER: %.2f| test loss: %.2f \n" % (test_WER, test_loss) )


def G2P(word):
	x = train_dataset.word_tokenizer.EncodeAsIds(word)
	T = [len(x)]
	x = torch.tensor(x).unsqueeze(0).long()
	y = model.infer(x,T)
	phonemes = train_dataset.phoneme_tokenizer.DecodeIds(y[0])
	return phonemes

# Test words as follows:
word = "wug"
print(word)
print(G2P(word))
