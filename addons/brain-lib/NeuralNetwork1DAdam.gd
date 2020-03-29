extends "NeuralNetwork1D.gd"

const NNNative = preload("res://bin/neural_network_1d_adam.gdns")

export(float, 0.0, 1.0, 0.0001) var learning_rate = 0.01 setget set_lr
export(float, 0.0, 1.0, 0.0001) var beta1 = 0.9 setget set_beta1
export(float, 0.0, 1.0, 0.0001) var beta2 = 0.999 setget set_beta2

func _ready():
	self._nn = NNNative.new()
	self._nn.learning_rate = self.learning_rate
	self._nn.beta1 = self.beta1
	self._nn.beta2 = self.beta2
	self.init_nn()

func set_lr(new_lr):
	learning_rate = new_lr
	if self._nn:
		self._nn.learning_rate = new_lr

func set_beta1(new_beta1):
	beta1 = new_beta1
	if self._nn:
		self._nn.beta1 = new_beta1

func set_beta2(new_beta2):
	beta2 = new_beta2
	if self._nn:
		self._nn.beta2 = new_beta2

func _has_props(props):
	return props.type == "adam"