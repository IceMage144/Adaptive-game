from random import choice, random
import math

import torch

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

from Characters.AIs.QLAI import QLAI
import util

@exposed
class PerceptronQLAI(QLAI):
	"""
	Q-Learning AI implemented using UC Berkeley CS188 course's material
	(http://ai.berkeley.edu/home.html).

	The material uses an one-dimensional matrix of weights to approximate
	the Q-value of a single action, and uses Belman's equation to update them.
	"""
	def _ready(self):
		super(PerceptronQLAI, self)._ready()
	
	def init(self, params):
		super(PerceptronQLAI, self).init(params)
		if not (params["network_id"] is None):
			character_type = params["character_type"]
			network_id = params["network_id"]
			self.network_key = f"{character_type}_PerceptronQLAI_{network_id}"
		persisted_params = self.load_params()
		self.learning_weights = 2.0 * torch.rand(self.features_size) + 1.0
		if not (persisted_params is None):
			self.learning_weights = persisted_params.get("model_params")
			self.time = persisted_params.get("time")

	def end(self):
		persistence_dict = {
			"time": self.time,
			"model_params": self.learning_weights
		}
		self.save_params(persistence_dict)

	def get_info(self):
		return util.py2gdArray(self.learning_weights.tolist())

	def _get_q_value(self, state, action):
		features = self._get_features_after_action(state, action)
		return features @ self.learning_weights

	def _compute_value_from_q_values(self, state):
		legal_actions = self.parent.get_legal_actions(state)
		return max([self._get_q_value(state, action) for action in legal_actions])

	def _compute_action_from_q_values(self, state):
		legal_actions = self.parent.get_legal_actions(state)
		if random() < self.epsilon:
			return choice(legal_actions)
		max_val = -math.inf
		max_action_set = []
		for a in legal_actions:
			val = self._get_q_value(state, a)
			if val == max_val:
				max_action_set.append(a)
			elif val > max_val:
				max_action_set = [a]
				max_val = val
		return choice(max_action_set)

	def _update_weights(self, state, action, next_state, reward, last):
		next_state_action_value = self._compute_value_from_q_values(next_state)
		target = reward
		if not last:
			target += self.discount * next_state_action_value
		prediction = self._get_q_value(state, action)
		correction = target - prediction
		features = self._get_features(next_state)

		self.learning_weights += self.alpha * correction * features
		self.learning_weights /= self.learning_weights.norm()

		self.parent.logger.push("loss", math.fabs(correction))
		self.parent.logger.push("max_q_val", next_state_action_value.item())
		self.parent.logger.push("reward", reward)

	# Print some variables for debug here
	def _on_DebugTimer_timeout(self):
		print("------ PerceptronQLAI ------")
		super(PerceptronQLAI, self)._on_DebugTimer_timeout()
		# print(self.get_info())