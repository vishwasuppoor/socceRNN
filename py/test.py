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
lstm_data = np.load('lstm_test1_data.npy')
fcn_data = np.load('fcn_test1_data.npy')
prediction_indices = np.load('prediction_indices_test1.npy')
interactions = np.load('interactions_test1.npy')
num_sequences, seq_len, vec_len = lstm_data.shape
output = np.zeros((num_sequences, seq_len, 46))
output[:,0] = lstm_data[:,0,:46]

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--seq_len', type=int, default=1)
argparser.add_argument('--chunk_len', type=int, default=49)
argparser.add_argument('--batch_size', type=int, default=1)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
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


def test(seq_num, lstm_inp, fcn_inp, target, interaction, prediction_index):
	lstm_model.eval()
	fcn_model.eval()
	hidden = lstm_model.init_hidden(args.batch_size)
	if args.cuda:
		c, h = hidden
		c = c.cuda()
		h = h.cuda()
		hidden = (c,h)
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
			lstm_output[0,2*prediction_index[c]:2*prediction_index[c]+2] += fcn_output[-2:]
			lstm_output[0,-2:] += fcn_output[:2]
		else:
			lstm_output, hidden = lstm_model(lstm_inp[:,c,:], hidden)
			lstm_loss += criterion(lstm_output.view(args.batch_size, -1), target[:,c,:])
		output[seq_num,c+1,:] = lstm_output.cpu().detach().numpy()

	return lstm_loss, fcn_t_loss


lstm_model = torch.load('lstm.pt')
fcn_model = torch.load('fcn.pt')

criterion = nn.MSELoss()

if args.cuda:
	lstm_model.cuda()
	fcn_model.cuda()

start = time.time()

print("Testing...")
total_lstm_loss, total_fcn_loss = 0, 0
for idx in range(num_sequences):
	ll, fl = test(idx, *batch(idx, args.batch_size))
	total_lstm_loss += ll
	total_fcn_loss += fl
print('testing loss [%.4f %.4f]' % (total_lstm_loss, total_fcn_loss))
np.save('test1_output.npy', output)
