extends "NeuralNetwork1D.gd"

const NNNative = preload("res://bin/neural_network_1d_adagrad.gdns")

export(float, 0.0, 1.0, 0.0001) var learning_rate = 0.01 setget set_lr

func _ready():
	self._nn = NNNative.new()
	self._nn.learning_rate = self.learning_rate
	self.init_nn()

func set_lr(new_lr):
	learning_rate = new_lr
	if self._nn:
		self._nn.learning_rate = new_lr

func _has_props(props):
	return props.type == "adagrad"