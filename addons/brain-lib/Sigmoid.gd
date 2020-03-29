extends Node

const Sigmoid = preload("res://bin/sigmoid.gdns")

var _layer

func _ready():
	self._layer = Sigmoid.new()

func _has_props(props):
	return (props.type == "sigmoid")