extends Node

const Tanh = preload("res://bin/tanh.gdns")

var _layer

func _ready():
	self._layer = Tanh.new()

func _has_props(props):
	return (props.type == "tanh")