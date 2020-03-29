extends Node

const FullyConnected = preload("res://bin/fully_connected.gdns")

export(int, 1, 10000) var size = 1
export(bool) var has_bias = true

var _layer

func _ready():
	self._layer = FullyConnected.new()
	self._layer.has_bias = self.has_bias
	self._layer.size = self.size

func _has_props(props):
	var prop1 = (props.type == "fully_connected")
	var prop2 = (props.out_size == self.size)
	var prop3 = (props.has_bias == self.has_bias)
	return prop1 and prop2 and prop3