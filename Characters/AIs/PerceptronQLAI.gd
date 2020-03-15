extends "res://Characters/AIs/QLAI.gd"

# Features -> Array
# Reward   -> float
# State    -> Dict
# Action   -> int

const Experience = preload("res://Characters/AIs/Experience.gd")
const NeuralNetwork = preload("res://Characters/AIs/MultiNN.tscn")

var ep

var learning_weights = []

func init(params):
	.init(params)
	self.ep = Experience.new(self.experience_pool_size)
	if params.has("network_id") and params.network_id != null:
		var character_type = params.character_type
		var network_id = params.network_id
		self.network_key = character_type + "_PerceptronQLAINative_" + network_id
	for i in range(self.features_size):
		self.learning_weights.append(2.0 * randf() + 1.0)
	# persisted_params = self.load_params()
	# if not (persisted_params is None):
	# 	self.learning_weights = persisted_params.get("model_params")
	# 	self.time = persisted_params.get("time")

func end():
	# persistence_dict = {
	# 	"time": self.time,
	# 	"model_params": self.learning_weights
	# }
	# self.save_params(persistence_dict)
	pass

func get_info():
	# return util.py2gdArray(self.learning_weights.tolist())
	pass

func _get_q_value(state, action):
	var out = self._get_features_after_action(state, action)
	for i in range(out.size()):
		out[i] *= self.learning_weights[i]
	return global.sum(out)

func _compute_value_from_q_values(state):
	var legal_actions = self.parent.get_legal_actions(state)
	var q_values = []
	for action in legal_actions:
		q_values.append(self._get_q_value(state, action))
	return global.max(q_values)

func _compute_action_from_q_values(state):
	var legal_actions = self.parent.get_legal_actions(state)
	if randf() < self.epsilon:
		return global.choose_one(legal_actions)
	var max_val = -INF
	var max_action_set = []
	for action in legal_actions:
		var val = self._get_q_value(state, action)
		if val == max_val:
			max_action_set.append(action)
		elif val > max_val:
			max_action_set = [action]
			max_val = val
	return global.choose_one(max_action_set)

func _update_weights(state, action, next_state, reward, last):
	var next_val = self._compute_value_from_q_values(next_state)
	var target = reward
	if not last:
		target += self.discount * next_val
	var prediction = self._get_q_value(state, action)
	var correction = target - prediction
	var features = self._get_features(next_state)

	for i in range(self.learning_weights.size()):
		self.learning_weights[i] += self.alpha * correction * features[i]
	global.normalize(self.learning_weights)

	self.parent.logger.push("loss", abs(correction))
	self.parent.logger.push("max_q_val", next_val)
	self.parent.logger.push("reward", reward)

# Print some variables for debug here
func _on_DebugTimer_timeout():
	print("------ PerceptronQLAI ------")
	._on_DebugTimer_timeout()
	# print(self.get_info())