import gym
import ptan
import argparse
import numpy as np
import os.path

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from lib import dqn_model, common


DEFAULT_N_STEP = 2
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100

def calc_values_of_states(states, net, device= "cpu"):
	mean_vals= []
	for batch in np.array_split(states, 64):
		states_v = torch.tensor(batch).to(device)
		action_values_v = net(states_v)
		best_action_values_v = action_values_v.max(1)[0]
		mean_vals.append(best_action_values_v.mean().item())
	return np.mean(mean_vals)

def save_model(model, modelfile):
	print("Saving model in {}".format(modelfile))
	torch.save(model.state_dict(), modelfile)

def load_model(model, modelfile):
	if os.path.isfile(modelfile):
		print("Loading model from {}".format(modelfile))
		model.load_state_dict(torch.load(modelfile, map_location= lambda storage, loc: storage))


if __name__ == '__main__':
	params = common.HYPERPARAMS['pong']
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", default= False, action= "store_true", help= "Enable cuda")
	parser.add_argument("--n", default= DEFAULT_N_STEP, type= int , help= "number of steps to DQN")
	parser.add_argument("--double", default= False, action= "store_true" , help= "Enable double DQN")
	parser.add_argument("-m", "--model", required = True, help= "Model file to load and save")

	args = parser.parse_args()
	device = torch.device("cuda" if args.cuda else "cpu")

	env = gym.make(params['env_name'])
	env = ptan.common.wrappers.wrap_dqn(env)

	writer = SummaryWriter("../../logs/07/03_dqn_double", comment= "-" + params['run_name'] + "-double")
	
	net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
	load_model(net, args.model)
	tgt_net = ptan.agent.TargetNet(net)

	selector = ptan.actions.EpsilonGreedyActionSelector(epsilon= params['epsilon_start'])
	epsilon_tracker = common.EpsilonTracker(selector, params)
	agent = ptan.agent.DQNAgent(net, selector, device= device)

	exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma= params['gamma'], steps_count= args.n)
	buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size= params['replay_size'])

	optimizer = optim.Adam(net.parameters(), lr= params['learning_rate'])
	frame_idx = 0
	eval_states = None

	with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
		while True:
			frame_idx += 1
			buffer.populate(1)
			epsilon_tracker.frame(frame_idx)
			new_rewards = exp_source.pop_total_rewards()
			if new_rewards:
				if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
					break
			if len(buffer) < params['replay_initial']:
				continue

			if eval_states is None:
				eval_states = buffer.sample(STATES_TO_EVALUATE)
				eval_states = [np.array(transition.state, copy= False) for transition in eval_states]
				eval_states = np.array(eval_states, copy=False)


			optimizer.zero_grad()
			batch = buffer.sample(params['batch_size'])
			loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, 
											gamma= params['gamma']**args.n, device= device,
											double = args.double)	
			loss_v.backward()
			optimizer.step()

			if frame_idx % params['target_net_sync'] == 0:
				tgt_net.sync()
				save_model(tgt_net.target_model, args.model)

			if frame_idx % EVAL_EVERY_FRAME == 0:
				mean_val = calc_values_of_states(eval_states, net, device= device)
				writer.add_scalar("values_mean", mean_val, frame_idx)


