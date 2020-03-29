extends Node

var Util = preload("Util.gd").new()
var CompositeType = Util.CompositeType

enum LOSS_TYPE {
	MINIMUM_SQUARES_ERROR,
	ABSOLUTE,
#	ABSOLUTE_EPS,
	CROSS_ENTROPY,
	CROSS_ENTROPY_MULTICLASS
}

const SAME_AS_INPUT = -1

var VALUE = CompositeType.new([TYPE_INT, TYPE_REAL])
var INPUT_VEC = CompositeType.new(TYPE_ARRAY)
var OUTPUT_VEC = CompositeType.new(TYPE_ARRAY)
var SEQUENCE = CompositeType.new(TYPE_ARRAY)
var ARRAY = CompositeType.new(TYPE_ARRAY)

var DATA_SET_TYPE = [ARRAY, INPUT_VEC, VALUE]
var SEQ_DATA_SET_TYPE = [ARRAY, SEQUENCE, INPUT_VEC, VALUE]
var OUT_SET_TYPE = [ARRAY, OUTPUT_VEC, VALUE]
var SEQ_OUT_SET_TYPE = [ARRAY, SEQUENCE, OUTPUT_VEC, VALUE]
var INPUT_TYPE = [INPUT_VEC, VALUE]

export(int, 1, 10000) var input_size = 1 setget set_input_size
export(LOSS_TYPE) var loss_func = MINIMUM_SQUARES_ERROR setget set_loss_func

var output_size
var depth
var _nn

func init_nn():
	self._nn.input_size = self.input_size
	self._nn.loss_function = self.loss_func

	self._nn.init()
	for child in self.get_children():
		self._nn.add_layer(child._layer)
	self._nn.init_layers()

	self.depth = self._nn.depth
	self.output_size = self._nn.output_size

	INPUT_VEC.size = self.input_size
	OUTPUT_VEC.size = self.output_size
	SEQUENCE.size = self.depth

func set_input_size(new_size):
	input_size = new_size
	if new_size != input_size and self._nn:
		self._nn.input_size = new_size

func set_loss_func(new_func):
	loss_func = new_func
	if new_func != loss_func and self._nn:
		self._nn.loss_func = new_func

func clear_memory():
	self._nn.clear_memory()

func save():
	Util.assert(
		self._nn.has_initialized(),
		"Network was not initialized yet, save result is inconsistent."
	)
	var attr_dict = self._nn.serialize_network()
	var parse_result = JSON.parse(attr_dict.model)
	Util.assert(
		parse_result.error == OK,
		"Model is inconsistent, cannot save."
	)
	attr_dict.model = parse_result.result.nodes
	return JSON.print(attr_dict)

func load(data):
	var parse_result = JSON.parse(data)
	Util.assert(
		parse_result.error == OK,
		"Could not load model from string, JSON is inconsistent."
	)
	parse_result = parse_result.result
	Util.assert(
		self._has_props(parse_result.optim),
		"Neural network does not have the required properties to be loaded."
	)
	var nodes_data = parse_result.model
	var children = self.get_children()
	Util.assert(
		nodes_data.size() == children.size(),
		"Network data size doesn't match tree size."
	)
	for i in range(nodes_data.size()):
		Util.assert(
			children[i]._has_props(nodes_data[i]),
			"Node " + children[i].name + " does not have the required properties to be loaded."
		)
	self._nn.load(parse_result.weights, parse_result.optim)

func forward(input):
	Util.assert(
		Util.is_compose_type(input, INPUT_TYPE),
		"NeuralNetwork1D.forward arg must be an Array[" + str(self.input_size) + "] of floats (input vector)."
	)
	return self._nn.forward(input)

func predict_one(input):
	Util.assert(
		Util.is_compose_type(input, INPUT_TYPE),
		"NeuralNetwork1D.predict_one arg must be an Array[" + str(self.input_size) + "] of floats (input vector)."
	)
	return self._nn.predict_one(input)

func predict(data_set):
	if self._nn.has_memory():
		return self._predict_sequence(data_set)
	return self._predict(data_set)

func train(data_set, out_set, batch_size = SAME_AS_INPUT, epoch = 1):
	Util.assert(
		data_set.size() == out_set.size(),
		"NeuralNetwork1D.train data set and out set sizes don't match."
	)
	Util.assert(
		data_set.size() >= batch_size,
		"NeuralNetwork1D.train batch size is greater than input Array."
	)
	if batch_size == SAME_AS_INPUT:
		batch_size = data_set.size()
	if self._nn.has_memory():
		return self._train_sequence(data_set, out_set, batch_size, epoch)
	return self._train(data_set, out_set, batch_size, epoch)

func loss(data_set, out_set):
	Util.assert(
		data_set.size() == out_set.size(),
		"NeuralNetwork1D.train data set and out set sizes don't match."
	)
	if self._nn.has_memory():
		return self._nn.loss_sequence(data_set, out_set)
	return self._nn.loss(data_set, out_set)

func _predict(data_set):
	Util.assert(
		Util.is_compose_type(data_set, DATA_SET_TYPE),
		"NeuralNetwork1D.predict arg must be an Array[] of Array[" + str(self.input_size) + "] of floats (array of input vectors)."
	)
	return self._nn.predict(data_set)

func _predict_sequence(data_set):
	Util.assert(
		Util.is_compose_type(data_set, SEQ_DATA_SET_TYPE),
		"NeuralNetwork1D.predict arg must be an Array[] of Array[" + str(self.depth) + "] of Array[" + str(self.input_size) + "] of floats (array of sequences of input vectors)."
	)
	return self._nn.predict_sequence(data_set)

func _train(data_set, out_set, batch_size, epoch):
	Util.assert(
		Util.is_compose_type(data_set, DATA_SET_TYPE),
		"NeuralNetwork1D.train first arg must be an Array[] of Array[" + str(self.input_size) + "] of floats (array of input vectors)."
	)
	Util.assert(
		Util.is_compose_type(out_set, OUT_SET_TYPE),
		"NeuralNetwork1D.train second arg must be an Array[] of Array[" + str(self.output_size) + "] of floats (array of output vectors)."
	)
	return self._nn.train(data_set, out_set, batch_size, epoch)

func _train_sequence(data_set, out_set, batch_size, epoch):
	Util.assert(
		Util.is_compose_type(data_set, SEQ_DATA_SET_TYPE),
		"NeuralNetwork1D.train first arg must be an Array[] of Array[" + str(self.depth) + "] of Array[" + str(self.input_size) + "] of floats (array of sequences of input vectors)."
	)
	Util.assert(
		Util.is_compose_type(out_set, SEQ_OUT_SET_TYPE),
		"NeuralNetwork1D.train second arg must be an Array[] of Array[" + str(self.depth) + "] of Array[" + str(self.output_size) + "] of floats (array of sequences of input vectors)."
	)
	return self._nn.train_sequence(data_set, out_set, batch_size, epoch)

func _loss(data_set, out_set):
	Util.assert(
		Util.is_compose_type(data_set, DATA_SET_TYPE),
		"NeuralNetwork1D.loss first arg must be an Array[] of Array[" + str(self.input_size) + "] of floats (array of input vectors)."
	)
	Util.assert(
		Util.is_compose_type(out_set, OUT_SET_TYPE),
		"NeuralNetwork1D.loss second arg must be an Array[] of Array[" + str(self.output_size) + "] of floats (array of output vectors)."
	)
	return self._nn.loss(data_set, out_set)

func _loss_sequence(data_set, out_set):
	Util.assert(
		Util.is_compose_type(data_set, SEQ_DATA_SET_TYPE),
		"NeuralNetwork1D.loss first arg must be an Array[] of Array[" + str(self.depth) + "] of Array[" + str(self.input_size) + "] of floats (array of sequences of input vectors)."
	)
	Util.assert(
		Util.is_compose_type(out_set, SEQ_OUT_SET_TYPE),
		"NeuralNetwork1D.loss second arg must be an Array[] of Array[" + str(self.depth) + "] of Array[" + str(self.output_size) + "] of floats (array of sequences of input vectors)."
	)
	return self._nn.loss_sequence(data_set, out_set)

# Virtual
func _has_props(props):
	return true