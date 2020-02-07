from random import choice, random

import torch
import torch.nn as nn

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

from Characters.AIs.QLAI import QLAI
from Characters.AIs.structs import Experience
import util


class DQN(nn.Module):
	def __init__(self, arch, learning_rate, weight_decay, model_params=None):
		super(DQN, self).__init__()
		self.layers = len(arch) - 1
		modules = []
		for i in range(self.layers):
			modules.append(nn.Linear(arch[i], arch[i+1], bias=False))
		self.model = nn.ModuleList(modules)
		if not (model_params is None):
			self.model.load_state_dict(model_params)
		self.activ = nn.Tanh()
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,
										  weight_decay=weight_decay)
    
	def forward(self, features):
		output = features
		for i in range(self.layers-1):
			output = self.activ(self.model[i](output))
		output = self.model[self.layers-1](output)
		return output
 
	def back(self, predicted, label):
		self.optimizer.zero_grad()
		loss = self.criterion(predicted, label)
		loss.backward()
		self.optimizer.step()
		return loss

	def predict(self, features):
		return torch.argmax(self.forward(features)).item()

@exposed
class SingleQLAI(QLAI):
	"""
	Q-Learning AI implemented using PyTorch.

	This implementation uses a neural network to approximate the Q-value of
	a single action, and uses adaptive gradient descent to update its weights.
	"""
	def _ready(self):
		super(SingleQLAI, self)._ready()
	
	def init(self, params):
		super(SingleQLAI, self).init(params)
		if not (params["network_id"] is None):
			character_type = params["character_type"]
			network_id = params["network_id"]
			self.network_key = f"{character_type}_SingleQLAI_{network_id}"
		persisted_params = self.load_params()
		model_params = None
		if not (persisted_params is None):
			model_params = persisted_params.get("model_params")
			self.time = persisted_params.get("time")
		self.learning_model = DQN([self.features_size, 16, 1], self.alpha,
								  0.01, model_params=model_params)
	
	def end(self):
		persistence_dict = {
			"time": self.time,
			"model_params": self.learning_model.model.state_dict()
		}
		self.save_params(persistence_dict)

	def get_info(self):
		# TODO: Use state_dict method
		return util.py2gdArray([param.tolist() for param in self.learning_model.parameters()])

	def reset(self, timeout):
		super(SingleQLAI, self).reset(timeout)
		if self.use_experience_replay and self.learning_activated:
			exp_sample = self.ep.sample()
			if not (exp_sample is None):
				loss = self._update_weights_experience(exp_sample)
				self.parent.logger.push("loss", loss.item())

	def _get_q_value(self, state, action):
		features = self._get_features_after_action(state, action)
		return self.learning_model.forward(features)[0]

	def _compute_value_from_q_values(self, state):
		if state is None:
			return torch.tensor(0.0)
		legal_actions = self.parent.get_legal_actions(state)
		features_list = torch.stack([self._get_features_after_action(state, a) for a in legal_actions])
		return torch.max(self.learning_model.forward(features_list))

	def _compute_action_from_q_values(self, state):
		legal_actions = self.parent.get_legal_actions(state)
		if random() < self.epsilon:
			return choice(legal_actions)
		features_list = torch.stack([self._get_features_after_action(state, a) for a in legal_actions])
		q_vals = self.learning_model.forward(features_list)
		prediction = torch.argmax(q_vals).item()
		return legal_actions[prediction]

	def _update_weights(self, state, action, next_state, reward, last):
		features = self._get_features(next_state)
		experience = Experience(features, reward, None if last else next_state)
		self.ep.add(experience)
		exp_sample = self.ep.simple_sample()
		self._update_weights_experience(exp_sample)
		self.parent.logger.push("reward", reward)

	def _update_weights_experience(self, exp_sample):
		actual_val_vec = []
		next_val_vec = []
		reward_vec = []

		for exp in exp_sample:
			actual_val_vec.append(self.learning_model.forward(exp.features)[0])

			next_val = self._compute_value_from_q_values(exp.next_state)
			next_val_vec.append(next_val)

			reward_vec.append(exp.reward)

		actual_val_vec = torch.stack(actual_val_vec)
		next_val_vec = torch.stack(next_val_vec)
		reward_vec = torch.tensor(reward_vec)
		label_vec = reward_vec + self.discount * next_val_vec

		return self.learning_model.back(actual_val_vec, label_vec)

	# Print some variables for debug here
	def _on_DebugTimer_timeout(self):
		print("------ SingleQLAI ------")
		super(SingleQLAI, self)._on_DebugTimer_timeout()
		# print(self.get_info())
