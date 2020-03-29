extends Node

const LSTM = preload("res://bin/lstm.gdns")

export(int, 1, 10000) var size = 1
export(int, 1, 10000) var depth = 2
export(int, 1, 10000) var num_layers = 1
export(float, 0.0, 1.0, 0.0001) var dropout_rate = 0.0
export(bool) var has_bias = true

var _layer

func _ready():
	self._layer = LSTM.new()
	self._layer.has_bias = self.has_bias
	self._layer.num_layers = self.num_layers
	self._layer.dropout_rate = self.dropout_rate
	self._layer.depth = self.depth
	self._layer.size = self.size

func _has_props(props):
	var prop1 = (props.type == "recurrent_layer")
	var prop2 = (props.seq_len == self.depth)
	var prop3 = (props.lstm_cell.out_size == self.size)
	var prop4 = (props.lstm_cell.has_bias == self.has_bias)
	return prop1 and prop2 and prop3 and prop4
