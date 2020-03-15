extends "res://Characters/AIs/QLAI.gd"

# Features -> Array
# Reward   -> float
# State    -> Dict
# Action   -> int

var Action = preload("res://Characters/ActionBase.gd").new()
const Experience = preload("res://Characters/AIs/Experience.gd")
const NeuralNetwork = preload("res://Characters/AIs/MultiNN.tscn")

var ep
var learning_model

var action_to_id = {}
var id_to_action = {}

onready var AVAILABLE_ACTIONS = [
	Action.from_string("idle"),
	Action.from_string("attack"),
	Action.from_string("walk_right"),
	Action.from_string("walk_up_right"),
	Action.from_string("walk_up"),
	Action.from_string("walk_up_left"),
	Action.from_string("walk_left"),
	Action.from_string("walk_down_left"),
	Action.from_string("walk_down"),
	Action.from_string("walk_down_right")
]

func _ready():
	for i in range(AVAILABLE_ACTIONS.size()):
		self.action_to_id[AVAILABLE_ACTIONS[i]] = i
		self.id_to_action[i] = AVAILABLE_ACTIONS[i]

# Dict -> void
func init(params):
	.init(params)
	self.ep = Experience.new(self.experience_pool_size)
	if params.has("network_id") and params.network_id != null:
		var character_type = params.character_type
		var network_id = params.network_id
		self.network_key = character_type + "_MultiQLAINative_" + network_id
	self.learning_model = NeuralNetwork.instance()
	self.learning_model.learning_rate = self.alpha
	self.learning_model.input_size = self.features_size
	self.learning_model.get_node("FullyConnected2").size = self.action_to_id.size()
	self.add_child(self.learning_model)

# -> void
func end():
	pass

# -> void
func get_info():
	# return util.py2gdArray([param.tolist() for param in self.learning_model.parameters()])
	pass

# bool -> void
func reset(timeout):
	.reset(timeout)
	if self.use_experience_replay and self.learning_activated:
		var exp_sample = self.ep.sample()
		self._update_weights_experience(exp_sample[0], exp_sample[1], exp_sample[2], exp_sample[3])

# State, Array[Action] -> Array[float]
func _get_q_values(state, action_list):
	var features = self._get_features(state)
	var output = self.learning_model.predict_one(features)
	var q_values = []
	for action in action_list:
		q_values.append(output[self.action_to_id[action]])
	return q_values

# State -> float
func _compute_value_from_q_values(state):
	if state == null:
		return 0.0
	var legal_actions = self.parent.get_legal_actions(state)
	return global.max(self._get_q_values(state, legal_actions))

# State -> Action
func _compute_action_from_q_values(state):
	var legal_actions = self.parent.get_legal_actions(state)
	if randf() < self.epsilon:
		return global.choose_one(legal_actions)
	var prediction = self._get_q_values(state, legal_actions)
	var best_action = legal_actions[global.argmax(prediction)]
	return best_action

# State, Action, State, Reward, bool -> void
func _update_weights(state, action, next_state, reward, last):
	var features = self._get_features(next_state)
	var action_id = self.action_to_id[action]
	if not last:
		self.ep.push(features, reward, next_state, action_id)
	var exp_sample = self.ep.simple_sample()
	self._update_weights_experience(exp_sample[0], exp_sample[1], exp_sample[2], exp_sample[3])
	self.parent.logger.push("reward", reward)

# Array[Features], Array[Reward], Array[State], Array[Action] -> void
func _update_weights_experience(feat_sample, reward_sample, next_sample, action_sample):
	var label_vec = []

	for i in range(feat_sample.size()):
		var next_val = self._compute_value_from_q_values(next_sample[i])
		var q_values = self.learning_model.predict_one(feat_sample[i])
		q_values[action_sample[i]] = reward_sample[i] + self.discount * next_val
		label_vec.append(q_values)
	
	self.learning_model.train(feat_sample, label_vec)

# Print some variables for debug here
func _on_DebugTimer_timeout():
	print("------ MultiQLAI ------")
	._on_DebugTimer_timeout()
	# print(self.get_info())
