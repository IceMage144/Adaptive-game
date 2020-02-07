from random import choice, random

import torch
import torch.nn as nn

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

from Characters.AIs.QLAI import QLAI
from Characters.AIs.structs import Experience
import util


class DQSN(nn.Module):
	def __init__(self, arch, memo_arch, learning_rate, weight_decay, model_params=None):
		super(DQSN, self).__init__()
		self.layers = len(arch) - 1
		modules = []
		for i in range(self.layers):
			modules.append(nn.Linear(arch[i], arch[i+1], bias=False))
		self.model = nn.ModuleList(modules)
		self.memo = nn.LSTM(arch[-1], memo_arch[0], num_layers=memo_arch[1], batch_first=True)
		self.last_layer = nn.Linear(memo_arch[0], memo_arch[2])
		if not (model_params is None):
			self.model.load_state_dict(model_params["model"])
			self.memo.load_state_dict(model_params["memo"])
			self.last_layer.load_state_dict(model_params["last_layer"])
		self.activ = nn.Tanh()
		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,
										  weight_decay=weight_decay)
		self.state = None
    
	def forward(self, features, set_internal_state=True):
		output = features
		for i in range(self.layers-1):
			output = self.activ(self.model[i](output))
		output = torch.stack([self.model[self.layers-1](output)])
		if output.dim() == 2:
			output = torch.stack([output])
		new_state = None
		if not (self.state is None):
			output, new_state = self.memo(output, self.state)
		else:
			output, new_state = self.memo(output)
		output = self.last_layer(output)
		if set_internal_state:
			self.state = new_state
		return output.view(-1)
 
	def back(self, predicted, label):
		self.optimizer.zero_grad()
		loss = torch.tanh(self.criterion(predicted, label))
		loss.backward()
		self.optimizer.step()
		return loss

	def predict(self, features):
		return torch.argmax(torch.cat([self.forward(item, set_internal_state=False) for item in features])).item()
	
	def reset_state(self):
		self.state = None
	
	def get_state(self):
		return self.state
	
	def set_state(self, new_state):
		self.state = new_state

@exposed
class MemoryQLAI(QLAI):
	"""
	Q-Learning AI implemented using PyTorch.

	This implementation uses a neural network with a LSTM to approximate
	the Q-values of a sequence of actions, and uses adaptive gradient descent
	to update its weights. The weights are updated every X calls to update_state.
	"""
	def _ready(self):
		super(MemoryQLAI, self)._ready()
		self.seq_size = 4
		self.seq_exp = Experience([], [], [])
	
	def init(self, params):
		super(MemoryQLAI, self).init(params)
		if not (params["network_id"] is None):
			character_type = params["character_type"]
			network_id = params["network_id"]
			self.network_key = f"{character_type}_MemoryQLAI_{network_id}"
		persisted_params = self.load_params()
		model_params = None
		if not (persisted_params is None):
			model_params = persisted_params.get("model_params")
			self.time = persisted_params.get("time")
		self.learning_model = DQSN([self.features_size, 9], [16, 2, 1], self.alpha,
									0.01, model_params=model_params)

	def end(self):
		persistence_dict = {
			"time": self.time,
			"model_params": {
				"model": self.learning_model.model.state_dict(),
				"memo": self.learning_model.memo.state_dict(),
				"last_layer": self.learning_model.last_layer.state_dict()
			}
		}
		self.save_params(persistence_dict)

	def get_info(self):
		# TODO: Use state_dict method
		return util.py2gdArray([param.tolist() for param in self.learning_model.parameters()])
	
	def reset(self, timeout):
		super(MemoryQLAI, self).reset(timeout)
		self.learning_model.reset_state()
		self.seq_exp = Experience([], [], [])
		if self.use_experience_replay and self.learning_activated:
			exp_sample = self.ep.sample()
			if not (exp_sample is None):
				loss = self._update_weights_experience(exp_sample)
				self.parent.logger.push("loss", loss.item())

	def _get_q_value(self, state, action):
		features = self._get_features_after_action(state, action)
		return self.learning_model(features, set_internal_state=False)

	def _compute_value_from_q_values(self, state):
		if state is None:
			return torch.tensor(0.0)
		legal_actions = self.parent.get_legal_actions(state)
		q_values_list = torch.cat([self._get_q_value(state, a) for a in legal_actions])
		return torch.max(q_values_list)

	def _compute_action_from_q_values(self, state):
		legal_actions = self.parent.get_legal_actions(state)
		if random() < self.epsilon:
			return choice(legal_actions)
		features_list = torch.stack([self._get_features_after_action(state, a) for a in legal_actions])
		return legal_actions[self.learning_model.predict(features_list)]

	def _update_weights(self, state, action, next_state, reward, last):
		features = self._get_features(next_state)
		self.seq_exp.append([features, reward, None if last else next_state])

		# FIXME: Wasting time just to update internal state
		self.learning_model(features)

		self.parent.logger.push("reward", reward)

		if len(self.seq_exp.features) >= self.seq_size:
			self.learning_model.reset_state()
			self.ep.add(self.seq_exp)
			exp_sample = self.ep.simple_sample()
			self._update_weights_experience(exp_sample)
			self.seq_exp = Experience([], [], [])

	def _update_weights_experience(self, exp_sample):
		actual_val_vec = []
		next_val_vec = []
		reward_vec = []
		for exp in exp_sample:
			actual_val_seq = [self.learning_model(features) for features in exp.features]
			actual_val_vec.append(torch.stack(actual_val_seq))

			next_val_seq = torch.tensor([self._compute_value_from_q_values(next_state) for next_state in exp.next_state])
			next_val_vec.append(next_val_seq.reshape((-1, 1)))

			reward_seq = torch.tensor(exp.reward)
			reward_vec.append(reward_seq.reshape((-1, 1)))

			self.learning_model.reset_state()
		actual_val_vec = torch.stack(actual_val_vec)
		next_val_vec = torch.stack(next_val_vec)
		reward_vec = torch.stack(reward_vec)
		label_vec = reward_vec + self.discount * next_val_vec
		return self.learning_model.back(actual_val_vec, label_vec)

	# Print some variables for debug here
	def _on_DebugTimer_timeout(self):
		print("------ MemoryQLAI ------")
		super(MemoryQLAI, self)._on_DebugTimer_timeout()
		# print(self.get_info())
