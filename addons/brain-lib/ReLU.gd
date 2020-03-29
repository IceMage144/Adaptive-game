extends Node

const ReLU = preload("res://bin/relu.gdns")

var _layer

func _ready():
	self._layer = ReLU.new()

func _has_props(props):
	return (props.type == "relu")