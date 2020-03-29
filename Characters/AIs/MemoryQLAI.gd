extends "res://Characters/AIs/QLAI.gd"

# Features -> Array[float]
# Reward   -> float
# State    -> Dict
# Action   -> int

const Experience = preload("res://Characters/AIs/Experience.gd")
const NeuralNetwork = preload("res://Characters/AIs/MemoryNN.tscn")

var ep
var learning_model

var seq_size = 4

# Dict -> void
func init(params):
	.init(params)
	self.ep = Experience.new(self.experience_pool_size, self.seq_size)
	if params.has("network_id") and params.network_id != null:
		var character_type = params.character_type
		var network_id = params.network_id
		self.network_key = character_type + "_MemoryQLAI_" + str(network_id)

	self.learning_model = NeuralNetwork.instance()
	self.learning_model.learning_rate = self.alpha
	self.learning_model.input_size = self.features_size
	self.learning_model.get_node("LSTM").depth = self.seq_size
	self.add_child(self.learning_model)
	
	var persisted_params = self.load_params()
	if persisted_params != null:
		self.learning_model.load(persisted_params.model)
		self.time = persisted_params.time

# -> void
func end():
	self.save_params({
		"model": self.learning_model.save(),
		"time": self.time
	})

# -> void
func get_info():
	return []

# bool -> void
func reset(timeout):
	.reset(timeout)
	self._clear_memory()
	if self.use_experience_replay and self.learning_activated:
		var exp_sample = self.ep.sample()
		self._update_weights_experience(exp_sample[0], exp_sample[1], exp_sample[2])

# -> void
func _clear_memory():
	self.ep.clean_end()
	self.learning_model.clear_memory()

# State, Action, bool -> float
func _get_q_value(state, action, change_memo=false):
	var features = self._get_features_after_action(state, action)
	if change_memo:
		return self.learning_model.forward(features)[0]
	return self.learning_model.predict_one(features)[0]

# State, bool -> float
func _compute_value_from_q_values(state, change_memo=false):
	if state == null:
		return 0.0
	var legal_actions = self.parent.get_legal_actions(state)
	var q_values_list = []
	for a in legal_actions:
		q_values_list.append(self._get_q_value(state, a, change_memo))
	return global.max(q_values_list)

# State -> Action
func _compute_action_from_q_values(state):
	var legal_actions = self.parent.get_legal_actions(state)
	if randf() < self.epsilon:
		return global.choose_one(legal_actions)
	var q_values_list = []
	for a in legal_actions:
		q_values_list.append(self._get_q_value(state, a))
	return legal_actions[global.argmax(q_values_list)]

# State, Action, State, Reward, bool -> void
func _update_weights(state, action, next_state, reward, last):
	var features = self._get_features(next_state)
	self.ep.push(features, reward, next_state)
	if last:
		self._clear_memory()
	else:
		# Just to change internal state
		self.learning_model.forward(features)

	if self.ep.end_is_clean():
		var exp_sample = self.ep.simple_sample()
		self._update_weights_experience(exp_sample[0], exp_sample[1], exp_sample[2])

# Array[Array[Features]], Array[Array[Reward]], Array[Array[State]] -> void
func _update_weights_experience(feat_sample, reward_sample, next_sample):
	var label_vec = []

	for i in range(feat_sample.size()):
		label_vec.append([])
		for j in range(self.seq_size):
			var next_val = self._compute_value_from_q_values(next_sample[i][j], true)
			var label = [reward_sample[i][j] + self.discount * next_val]
			label_vec[i].append(label)
	
	self.learning_model.train(feat_sample, label_vec)

# Print some variables for debug here
func _on_DebugTimer_timeout():
	print("------ MemoryQLAI ------")
	._on_DebugTimer_timeout()
