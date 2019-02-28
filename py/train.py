import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import numpy as np
import time
from tqdm import tqdm

from helpers import *
from model import *

# load the data
lstm_data = np.load('lstm_data.npy')
fcn_data = np.load('fcn_data.npy')
prediction_indices = np.load('prediction_indices.npy')
interactions = np.load('interactions.npy')
num_sequences, seq_len, vec_len = lstm_data.shape

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--n_epochs', type=int, default=100)
argparser.add_argument('--seq_len', type=int, default=1)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=512)
argparser.add_argument('--n_layers', type=int, default=1)
argparser.add_argument('--learning_rate', type=float, default=0.001)
argparser.add_argument('--chunk_len', type=int, default=49)
argparser.add_argument('--batch_size', type=int, default=1)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--save', action='store_true')
argparser.add_argument('--cont_train', action='store_true')
args = argparser.parse_args()

if args.cuda:
	print("Using CUDA")


def batch(start, batch_size):
	lstm_inp = torch.from_numpy(lstm_data[start:start+batch_size,:-1])
	fcn_inp = torch.from_numpy(fcn_data[start,:-1]).float()
	target = torch.from_numpy(lstm_data[start:start+batch_size,1:,:46])
	lstm_inp = Variable(lstm_inp)
	fcn_inp = Variable(fcn_inp)
	target = Variable(target)
	interaction = interactions[start,:-1]
	prediction_index = prediction_indices[start,:-1]
	if args.cuda:
		lstm_inp = lstm_inp.cuda()
		fcn_inp = fcn_inp.cuda()
		target = target.cuda()
	return lstm_inp, fcn_inp, target, interaction, prediction_index


def train(lstm_inp, fcn_inp, target, interaction, prediction_index):
	lstm_model.train()
	fcn_model.train()
	hidden = lstm_model.init_hidden(args.batch_size)
	if args.cuda:
		c, h = hidden
		c = c.cuda()
		h = h.cuda()
		hidden = (c,h)
	lstm_model.zero_grad()
	lstm_loss = 0
	fcn_t_loss = 0

	for c in range(args.chunk_len):
		if interaction[c] > 0:
			lstm_output, hidden = lstm_model(lstm_inp[:,c,:], hidden)
			int_lstm_output = torch.cat([lstm_output[:,:2*prediction_index[0]],lstm_output[:,2*prediction_index[0]+2:-2]], 1)
			int_target = torch.cat([target[:,c,:2*prediction_index[c]],target[:,c,2*prediction_index[c]+2:-2]], 1)
			lstm_loss += criterion(int_lstm_output.view(args.batch_size, -1), int_target)

			fcn_output = fcn_model(fcn_inp[c])
			fcn_target = torch.cat([target[0,c,-2:],target[0,c,2*prediction_index[c]:2*prediction_index[c]+2]])
			lstm_op = torch.cat([lstm_output[0,-2:],lstm_output[0,2*prediction_index[c]:2*prediction_index[c]+2]])
			fcn_loss = criterion(fcn_output, fcn_target - lstm_op)
			fcn_t_loss += fcn_loss
			fcn_model.zero_grad()
			fcn_loss.backward(retain_graph=True)
			fcn_optimizer.step()
		else:
			lstm_output, hidden = lstm_model(lstm_inp[:,c,:], hidden)
			lstm_loss += criterion(lstm_output.view(args.batch_size, -1), target[:,c,:])

	lstm_loss.backward()
	lstm_optimizer.step()

	return lstm_loss, fcn_t_loss


def save():
	lstm_filename = 'lstm.pt'
	fcn_filename = 'fcn.pt'
	torch.save(lstm_model, lstm_filename)
	torch.save(fcn_model, fcn_filename)
	print('Saved as %s, %s' % (lstm_filename, fcn_filename))


if args.cont_train:
	lstm_model = torch.load('lstm.pt')
	fcn_model = torch.load('fcn.pt')
else:
	lstm_model = SoccerRNN(
		args.seq_len,
		90,
		args.hidden_size,
		46,
		args.n_layers,
		0
	)
	fcn_model = SoccerFCN(
		[44, 512, 4]
	)

lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=args.learning_rate)
fcn_optimizer = torch.optim.Adam(fcn_model.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss()

if args.cuda:
	lstm_model.cuda()
	fcn_model.cuda()

start = time.time()

print("Training for %d epochs..." % args.n_epochs)
for epoch in tqdm(range(1, args.n_epochs + 1)):
	total_lstm_loss, total_fcn_loss = 0, 0
	for idx in range(num_sequences):
		ll, fl = train(*batch(idx, args.batch_size))
		total_lstm_loss += ll
		total_fcn_loss += fl
	print('[%s (%d)]' % (time_since(start), epoch))
	if epoch % args.print_every == 0:
		print('training loss [%.4f %.4f]' % (total_lstm_loss, total_fcn_loss))

if args.save:
	print("Saving...")
	save()
