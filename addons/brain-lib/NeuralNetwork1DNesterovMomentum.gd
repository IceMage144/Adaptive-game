extends "NeuralNetwork1D.gd"

const NNNative = preload("res://bin/neural_network_1d_nesterov.gdns")

export(float, 0.0, 1.0, 0.0001) var learning_rate = 0.01 setget set_lr
export(float, 0.0, 1.0, 0.0001) var weight_decay = 0.0 setget set_wd
export(float, 0.0, 1.0, 0.0001) var momentum = 0.9 setget set_momentum

func _ready():
	self._nn = NNNative.new()
	self._nn.learning_rate = self.learning_rate
	self._nn.weight_decay = self.weight_decay
	self.init_nn()

func set_lr(new_lr):
	learning_rate = new_lr
	if self._nn:
		self._nn.learning_rate = new_lr

func set_wd(new_wd):
	weight_decay = new_wd
	if self._nn:
		self._nn.wd = new_wd

func set_momentum(new_momentum):
	momentum = new_momentum
	if self._nn:
		self._nn.momentum = new_momentum

func _has_props(props):
	return props.type == "nesterov_momentum"