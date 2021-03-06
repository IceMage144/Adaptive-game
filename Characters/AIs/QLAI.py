from random import choice, random
import math
import time
import base64

import pickle
import torch

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

import util
import Characters.AIs.structs as structs

@exposed
class QLAI(Node):
	def _ready(self):
		self.add_to_group("has_arch")
		self.parent = self.get_parent()
		self.logger = util.Logger()
		self.time = 0.0

	def init(self, params):
		self.learning_activated = params["learning_activated"]
		self.alpha = params["learning_rate"]
		self.discount = params["discount"]
		self.epsilon = params["max_exploration_rate"]
		self.max_epsilon = params["max_exploration_rate"]
		self.min_epsilon = params["min_exploration_rate"]
		self.epsilon_decay_time = params["exploration_rate_decay_time"]
		self.use_experience_replay = params["experience_replay"]
		self.ep = structs.ExperiencePool(params["experience_pool_size"])
		self.think_time = params["think_time"]
		self.features_size = params["features_size"]
		self.last_state = params["initial_state"]
		self.last_action = params["initial_action"]
		self.debug_mode = params["debug_mode"]
		self.network_key = None
	
	def reset(self, timeout):
		self.last_state = self.parent.get_state()

	# Abstract
	def end(self):
		pass
	
	def load_params(self):
		if self.network_key is None:
			return None
		ascii_data = NNParamsManager.get_params(self.network_key)
		if ascii_data is None:
			return None
		bytes_data = base64.decodebytes(ascii_data.encode())
		return pickle.loads(bytes_data)
	
	def save_params(self, params):
		if self.network_key is None:
			return
		bytes_data = pickle.dumps(params)
		ascii_data = base64.encodebytes(bytes_data).decode()
		NNParamsManager.set_params(self.network_key, ascii_data)
	
	def get_loss(self):
		return util.py2gdArray(self.logger.get_stored("loss"))
	
	def get_name(self):
		return self.parent.name

	def get_action(self):
		return self.last_action

	def update_state(self, last=False, timeout=False):
		ts = time.time()

		state = self.parent.get_state()

		if self.learning_activated:
			self.time += self.think_time
			reward = self.parent.get_reward(self.last_state, state, timeout)
			self._update_weights(self.last_state, self.last_action, state, reward, last)

		self.last_action = self._compute_action_from_q_values(state)
		self.last_state = state
		
		self._update_epsilon()

		te = time.time()
		self.logger.push("update_state", (te - ts) * 1000)

	# Abstract
	def _compute_action_from_q_values(self, state):
		pass

	# Abstract
	def _update_weights(self, actual_state, action, next_state, reward, last):
		pass

	def _get_features_after_action(self, state, action):
		par_out = self.parent.get_features_after_action(state, action)
		return torch.tensor(par_out)

	def _get_features(self, state):
		par_out = self.parent.get_features(state)
		return torch.tensor(par_out)
	
	def _update_epsilon(self):
		if not self.epsilon_decay_time:
			self.epsilon = self.min_epsilon
		else:
			factor = min(self.time, self.epsilon_decay_time) / self.epsilon_decay_time
			self.epsilon = factor * self.min_epsilon + (1.0 - factor) * self.max_epsilon

	# Print some variables for debug here
	def _on_DebugTimer_timeout(self):
		pass
