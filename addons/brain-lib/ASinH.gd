extends Node

const ASinH = preload("res://bin/asinh.gdns")

var _layer

func _ready():
	self._layer = ASinH.new()

func _has_props(props):
	return (props.type == "asinh")