extends Node

var tag
var model

var _cached_data = {}

func _ready():
	tag = self._get_tag()
	model = self._get_model()
	# Assert that the inherited class sets an unique tag
	assert(tag != "")
	# Assert that the inherited class sets a model
	assert(model != null)
	self._cached_data = SaveManager.load_data(tag)
	# print(tag + " loaded " + self._print_debug(self._cached_data))
	for key in model:
		var data_schema = model[key]
		if data_schema.has("default") and not self._cached_data.has(key):
			# print("Overwrited data: " + str(key) + " with " + str(data_schema["default"]))
			self._cached_data[key] = data_schema["default"]

func _print_debug(value):
	if typeof(value) == TYPE_STRING:
		if value.length() > 30:
			return str(value.length()) + " characters long string"
		return value
	elif typeof(value) == TYPE_DICTIONARY:
		if value.size() > 17:
			return str(value.size()) + " keys dictionary"
		var res = {}
		for key in value.keys():
			res[key] = self._print_debug(value[key])
		return str(res)
	elif typeof(value) == TYPE_ARRAY:
		if value.size() > 20:
			return str(value.size()) + " elements array"
		var res = []
		for el in value:
			res.append(self._print_debug(el))
		return str(res)
	return str(value)

func has_data(key):
	return self._cached_data.has(key)

func get_data(key):
	return self._cached_data[key]

func set_data(key, value):
	# Assert that key exists
	assert(model.has(key))
	# Assert that the value is the right type
	assert(model[key].type == typeof(value))
	self._cached_data[key] = value
	SaveManager.save_data(tag, self._cached_data)
	# print(tag + " saved " + self._print_debug(self._cached_data))

# Abstract
func _get_tag():
	return ""

# Abstract
func _get_model():
	return null