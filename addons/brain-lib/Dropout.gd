extends Node

const Dropout = preload("res://bin/dropout.gdns")

export(float, 0.0, 1.0, 0.0001) var dropout_rate = 0.0

var _layer

func _ready():
	self._layer = Dropout.new()
	self._layer.dropout_rate = self.dropout_rate

func _has_props(props):
	var prop1 = (props.type == "dropout")
	var prop2 = (props.dropout_rate == self.dropout_rate)
	return prop1 and prop2