extends "res://Persistence/ModelBase.gd"

const DEFAULT_PARAMS_PATH = "res://assets/data/default_ais.json"

const PARAMS = "params"

func _get_tag():
	return "ia"

func _get_model():
	var model_file = File.new()
	model_file.open(DEFAULT_PARAMS_PATH, File.READ)
	var parse_res = JSON.parse(model_file.get_as_text())
	var default_params = {}
	if parse_res.error == OK:
		default_params = parse_res.result
	model_file.close()

	return {
		PARAMS: {
			"type": TYPE_DICTIONARY,
			"default": default_params
		}
	}

func get_params(key):
	var params_dict = self.get_data(PARAMS)
	if not params_dict.has(key):
		return null
	return params_dict[key]

func set_params(key, value):
	var params_dict = self.get_data(PARAMS)
	params_dict[key] = value
	print("Saved " + key + " params")
	self.set_data(PARAMS, params_dict)