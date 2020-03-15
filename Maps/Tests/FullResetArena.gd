extends Node2D

var arena_height
var arena_width

var debug_mode = false
var tile_size = 32

func _ready():
	self.arena_width = 27 * self.tile_size
	self.arena_height = 13 * self.tile_size
	for character in self.get_tree().get_nodes_in_group("character"):
		character.connect("character_death", self, "_on_character_death")
		var off = 2 * self.tile_size
		character.position = Vector2(off + self.arena_width * randf(), \
									 off + self.arena_height * randf())
		character.init({"network_id": 1})

func init(params):
	pass

func print_info():
	var loss_info = {}
	print("------------")
	for character in self.get_tree().get_nodes_in_group("robot"):
		team = global.get_team(character)
		print(character.get_pretty_name() + ": " + str(character.life) + " (" + team + ")")
		if not loss_info.has(team):
			loss_info[team] = {}
		loss_info[team][character.name] = character.controller.get_loss()
	
	# Plot loss

func reset(timeout):
	var characters = self.get_tree().get_nodes_in_group("character")
	for character in characters:
		character.before_reset(timeout)
	for character in characters:
		character.reset(timeout)
	for character in characters:
		character.after_reset(timeout)
	for character in characters:
		character.end()

	var main = global.find_entity("main")
	main.change_map("res://Maps/Tests/PersistenceArena.tscn")

func _on_character_death():
	self.reset(false)

func _on_TimeoutTimer_timeout():
	self.reset(true)
