extends Node

var learning_activated
var alpha
var discount
var epsilon
var max_epsilon
var min_epsilon
var epsilon_decay_time
var use_experience_replay
var experience_pool_size
var think_time
var features_size
var last_state
var last_action
var debug_mode

var network_key = null
var time = 0.0

onready var parent = self.get_parent()

func _ready():
	self.add_to_group("has_arch")

func init(params):
	self.learning_activated = params["learning_activated"]
	self.alpha = params["learning_rate"]
	self.discount = params["discount"]
	self.epsilon = params["max_exploration_rate"]
	self.max_epsilon = params["max_exploration_rate"]
	self.min_epsilon = params["min_exploration_rate"]
	self.epsilon_decay_time = params["exploration_rate_decay_time"]
	self.use_experience_replay = params["experience_replay"]
	self.experience_pool_size = params["experience_pool_size"]
	self.think_time = params["think_time"]
	self.features_size = params["features_size"]
	self.last_state = params["initial_state"]
	self.last_action = params["initial_action"]
	self.debug_mode = params["debug_mode"]

func reset(timeout):
	self.last_state = self.parent.get_state()

# Abstract
func end():
	pass

func load_params():
	# if self.network_key is None:
	# 	return None
	# ascii_data = NNParamsManager.get_params(self.network_key)
	# if ascii_data is None:
	# 	return None
	# bytes_data = base64.decodebytes(ascii_data.encode())
	# return pickle.loads(bytes_data)
	pass

func save_params(params):
	# if self.network_key == null:
	# 	return
	# var bytes_data = pickle.dumps(params)
	# var ascii_data = base64.encodebytes(bytes_data).decode()
	# NNParamsManager.set_params(self.network_key, ascii_data)
	pass

func get_loss():
	# return util.py2gdArray(self.logger.get_stored("loss"))
	pass

func get_name():
	return self.parent.name

func get_action():
	return self.last_action

func update_state(state, last=false, timeout=false):
	if self.learning_activated:
		self.time += self.think_time
		var reward = self.parent.get_reward(self.last_state, state, timeout)
		self._update_weights(self.last_state, self.last_action, state, reward, last)

	self.last_action = self._compute_action_from_q_values(state)
	self.last_state = state
	
	self._update_epsilon()

# Abstract
func _compute_action_from_q_values(state):
	pass

# Abstract
func _update_weights(actual_state, action, next_state, reward, last):
	pass

func _get_features_after_action(state, action):
	return self.parent.get_features_after_action(state, action)

func _get_features(state):
	return self.parent.get_features(state)

func _update_epsilon():
	if not self.epsilon_decay_time:
		self.epsilon = self.min_epsilon
	else:
		var factor = min(self.time, self.epsilon_decay_time) / self.epsilon_decay_time
		self.epsilon = factor * self.min_epsilon + (1.0 - factor) * self.max_epsilon

# Print some variables for debug here
func _on_DebugTimer_timeout():
	var stats = ["max", "min", "avg"]
	self.parent.logger.print_stats("update_state", stats)
	# self.logger.print_stats("max_q_val", stats)
	# self.logger.print_stats("reward", stats)
	self.parent.logger.flush("update_state")
	# self.logger.flush("max_q_val")
	# self.logger.flush("reward")
	# print("Max weight: ", util.apply_list_func(self.get_info(), max))
	# print("Min weight: ", util.apply_list_func(self.get_info(), min))
	# print("epsilon: {}".format(self.epsilon))
