from lib import wrappers
from lib import dqn_model

import gym
import argparse
import time
import numpy as np 

import torch

import collections
import os


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

def configure_output_dir(d=None):
    """
    Set output directory to d, or to /tmp/somerandomnumber if d is None
    """
    G.output_dir = d or "./tmp/experiments/%i"%int(time.time())
    assert not osp.exists(G.output_dir), "Log dir %s already exists! Delete it first or use a different dir"%G.output_dir
    os.makedirs(G.output_dir)
    G.output_file = open(osp.join(G.output_dir, "log.txt"), 'w')
    atexit.register(G.output_file.close)
    print(colorize("Logging data to %s"%G.output_file.name, 'green', bold=True))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", required = True, help= "Model file to load")
	parser.add_argument("-e", "--env", default= DEFAULT_ENV_NAME, 
						help= "Name of the enviroment, default=" + DEFAULT_ENV_NAME)
	parser.add_argument("-r", "--record", 
						help= "Directory to store video recording")
	parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
	args = parser.parse_args()

	env = wrappers.make_env(args.env)

	if args.record:
		if 'env' in vars(args):
			env = gym.wrappers.Monitor(env, args.record + '/' + args.env + str(time.time()) + '/')
		else:
			env = gym.wrappers.Monitor(env, args.record + '/' + str(time.time()) + '/')

	net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
	net.load_state_dict(torch.load(args.model, map_location= lambda storage, loc: storage))

	state = env.reset()
	total_reward = 0
	c = collections.Counter()

	while True:

		start_ts = time.time()
		if args.visualize:
			env.render()
		state_v = torch.tensor(np.array([state], copy= False))
		q_vals =  net(state_v).data.numpy()[0]
		action = np.argmax(q_vals)
		c[action] += 1
		state, reward, done, _ = env.step(action)
		total_reward += reward
		if done:
			break
		delta = 1 / FPS - (time.time() - start_ts)
		if delta > 0:
			time.sleep(delta)

	print("Total reward: %.2f" % total_reward)
	print("Action counts:", c)
	if args.record:
		env.env.close()

		