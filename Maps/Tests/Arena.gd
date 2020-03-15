extends Node2D

const GRAPH_FREQUENCY = 10

var arena_width
var arena_height

var debug_mode = false
var initial_positions = {}
var tile_size = 32
var rounds = 0

func _ready():
	self.arena_width = 27 * self.tile_size
	self.arena_height = 13 * self.tile_size
	for character in self.get_tree().get_nodes_in_group("character"):
		self.initial_positions[character.name] = character.position
		character.connect("character_death", self, "_on_character_death", [character])
		character.init({})
		# character.init({"network_id": character.name})

func init(params):
	pass

func print_info():
	var loss_info = {}
	print("------------")
	for character in self.get_tree().get_nodes_in_group("robot"):
		var pretty_name = character.get_pretty_name()
		print(pretty_name + ": " + str(character.life))
		loss_info[pretty_name] = character.controller.get_loss()
	
	# if self.rounds % GRAPH_FREQUENCY == 0:
	# 	Plot loss

func reset(timeout):
	self.rounds += 1
	if self.debug_mode:
		self.print_info()
	# self.get_parent().reset_game()
	var characters = self.get_tree().get_nodes_in_group("character")
	for character in characters:
		character.before_reset(timeout)
	for character in characters:
		var off = 2 * self.tile_size
		var xPos = off + self.arena_width * randf()
		var yPos = off + self.arena_height * randf()
		character.position = Vector2(xPos, yPos)
		character.reset(timeout)
	for character in characters:
		character.after_reset(timeout)
	for character in characters:
		character.end()

func _on_character_death(character):
	$TimeoutTimer.start()
	print(character.name + " lost!")
	self.reset(false)

func _on_TimeoutTimer_timeout():
	self.reset(true)
	print("Timeout!")
