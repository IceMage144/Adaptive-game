extends Node

const SAVE_PATH = "user://save_data.json"

var cached_data = {}

func _ready():
	var save_file = File.new()
	if save_file.file_exists(SAVE_PATH):
		save_file.open(SAVE_PATH, File.READ)
		var parse_res = JSON.parse(save_file.get_as_text())
		# Assert that the JSON is not corrupted
		assert(parse_res.error == OK)
		self.cached_data = parse_res.result
	save_file.close()

func delete_data():
	self.cached_data = {}
	self.commit_data()
	print("Deleted save")

func save_data(key, value):
	self.cached_data[key] = value
	self.commit_data()

func load_data(key):
	if not self.cached_data.has(key):
		return {}
	return self.cached_data[key]

func commit_data():
	var save_file = File.new()
	save_file.open(SAVE_PATH, File.WRITE)
	save_file.store_string(JSON.print(self.cached_data, " "))
	save_file.close()
	print("Commited changes")

func has_save():
	return File.new().file_exists(SAVE_PATH)