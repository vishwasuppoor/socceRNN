import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import numpy as np

# load the data
lstm_data = np.load('lstm_test1_data.npy')[:,0,:]
initial_interactions = np.load('interactions_test1.npy')[:,0]
num_sequences, vec_len = lstm_data.shape
seq_len = 50
output = np.zeros((num_sequences, seq_len, 46))
output[:,0] = lstm_data[:,:46]

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


def initialize(start, batch_size):
	lstm_inp = torch.from_numpy(lstm_data[start:start+batch_size])
	# target = torch.from_numpy(lstm_data[start:start+batch_size,1:,:46])
	lstm_inp = Variable(lstm_inp)
	# target = Variable(target)
	interaction = initial_interactions[start]
	if args.cuda:
		lstm_inp = lstm_inp.cuda()
	return lstm_inp, interaction


def test(seq_num, lstm_inp, interaction):
	lstm_model.eval()
	fcn_model.eval()
	hidden = lstm_model.init_hidden(args.batch_size)
	if args.cuda:
		c, h = hidden
		c = c.cuda()
		h = h.cuda()
		hidden = (c,h)
	# lstm_loss = 0
	# fcn_t_loss = 0
	prev_prev_out = lstm_inp
	prev_out = lstm_inp
	ball_dist = lstm_inp[:,46:]
	flag = 1 if interaction > 0 else 0
	context = 10 / 34

	for c in range(args.chunk_len):
		if flag == 1:
			fcn_data = torch.zeros([44])
			vel = prev_out[0,:46] - prev_prev_out[0,:46]
			position_x = prev_out[0,:46:2]
			position_y = prev_out[0,1:46:2]
			ball_dist_x = ball_dist[0,::2]
			ball_dist_y = ball_dist[0,1::2]
			vel_x = vel[::2]
			vel_y = vel[1::2]
			player_indices = torch.arange(22)
			distances = torch.sqrt(ball_dist_x**2 + ball_dist_y**2)
			mask = (torch.abs(ball_dist_x) < context) * (torch.abs(ball_dist_y) < context)
			insta_ball_dist = distances[mask]
			insta_indices = np.argsort(insta_ball_dist.cpu().detach().numpy())
			insta_player_indices = player_indices[mask][insta_indices]
			prediction_index = insta_player_indices[0]
			insta_pos_x = position_x[:-1][mask][insta_indices]
			insta_pos_y = position_y[:-1][mask][insta_indices]
			insta_ball_x = ball_dist_x[mask][insta_indices]
			insta_ball_y = ball_dist_y[mask][insta_indices]
			insta_velocity_x = vel_x[:-1][mask][insta_indices]
			insta_velocity_y = vel_y[:-1][mask][insta_indices]
			fcn_data[0] = prev_out[0,44]
			fcn_data[1] = prev_out[0,45]
			fcn_data[2] = vel_x[-1]
			fcn_data[3] = vel_y[-1]
			fcn_data[4] = insta_pos_x[0]
			fcn_data[5] = insta_pos_y[0]
			fcn_data[6] = insta_velocity_x[0]
			fcn_data[7] = insta_velocity_y[0]
			fcn_data[8] = -1 if insta_player_indices[0] <= 10 else 1
			c = 1
			e = 9
			for d in range(1,len(insta_indices)):
				fcn_data[e] = insta_pos_x[d]
				fcn_data[e+1] = insta_pos_y[d]
				fcn_data[e+2] = insta_ball_x[d]
				fcn_data[e+3] = insta_ball_y[d]
				fcn_data[e+4] = insta_velocity_x[d]
				fcn_data[e+5] = insta_velocity_y[d]
				fcn_data[e+6] = -1 if insta_player_indices[d] <= 10 else 1
				e += 7
				c += 1
				if c == 6:
					break

			if args.cuda:
				prev_out = prev_out.cuda()
			lstm_output, hidden = lstm_model(prev_out, hidden)
			# int_lstm_output = torch.cat([lstm_output[:,:2*prediction_index],lstm_output[:,2*prediction_index+2:-2]], 1)
			# int_target = torch.cat([target[:,c,:2*prediction_index],target[:,c,2*prediction_index+2:-2]], 1)
			# lstm_loss += criterion(int_lstm_output.view(args.batch_size, -1), int_target)

			if args.cuda:
				fcn_data = fcn_data.cuda()
			fcn_output = fcn_model(fcn_data)
			# fcn_target = torch.cat([target[0,c,-2:],target[0,c,2*prediction_index:2*prediction_index+2]])
			# lstm_op = torch.cat([lstm_output[0,-2:],lstm_output[0,2*prediction_index:2*prediction_index+2]])
			# fcn_loss = criterion(fcn_output, fcn_target - lstm_op)
			# fcn_t_loss += fcn_loss
			lstm_output[0,2*prediction_index:2*prediction_index+2] += fcn_output[-2:]
			lstm_output[0,-2:] += fcn_output[:2]
		else:
			if args.cuda:
				prev_out = prev_out.cuda()
			lstm_output, hidden = lstm_model(prev_out, hidden)
			# lstm_loss += criterion(lstm_output.view(args.batch_size, -1), target[:,c,:])
		output[seq_num,c+1,:] = lstm_output.cpu().detach().numpy()
		prev_prev_out = prev_out.cpu()
		prev_out = torch.zeros([1,90])
		prev_out[:,:46] = lstm_output
		ball_dist = torch.zeros([1,44])
		ball_dist[:,::2] = (prev_out[:,:46][:,:-2:2] - prev_out[0,:46][-2])
		ball_dist[:,1::2] = (prev_out[:,:46][:,1:-2:2] - prev_out[0,:46][-1])
		prev_out[:,46:] = ball_dist
		flag = 1 if torch.sum((torch.abs(prev_out[0,46:][::2]) <= 0.001) * (torch.abs(prev_out[0,46:][1::2]) <= 0.001)) > 0 else 0

	# return lstm_loss, fcn_t_loss


lstm_model = torch.load('lstm.pt')
fcn_model = torch.load('fcn.pt')

criterion = nn.MSELoss()

if args.cuda:
	lstm_model.cuda()
	fcn_model.cuda()


print("Testing...")
# total_lstm_loss, total_fcn_loss = 0, 0
for idx in range(num_sequences):
	test(idx, *initialize(idx, args.batch_size))
	# total_lstm_loss += ll
	# total_fcn_loss += fl
# print('testing loss [%.4f %.4f]' % (total_lstm_loss, total_fcn_loss))
np.save('test1_output_.npy', output)
