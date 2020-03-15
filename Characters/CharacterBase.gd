extends KinematicBody2D

const PlayerController = preload("res://Characters/Player/PlayerController.gd")
const ControllerNode = preload("res://Characters/Controller.tscn")
const ActionClass = preload("res://Characters/ActionBase.gd")

signal character_death

enum Controller { PLAYER, AI }
enum AIType { PERCEPTRON, SINGLE, MEMORY, MULTI }

const KNOCKBACK_INITIAL_SPEED = 800
const KNOCKBACK_DECAY_RATE = 0.6
const KNOCKBACK_MIN = Vector2(20, 20)

const ai_name = ["Perceptron", "Single", "Memory", "Multi"]

export(int) var speed = 120
export(float) var weight = 1
export(int) var max_life = 3
export(int) var damage = 1
export(int) var defense = 0
export(Controller) var controller_type = Controller.PLAYER
export(AIType) var ai_type = AIType.PERCEPTRON
export(bool) var learning_activated = true
export(float, 0.0, 1.0, 0.0001) var learning_rate = 0.0
export(float, 0.0, 1.0, 0.001) var discount = 0.0
export(float, 0.0, 1.0, 0.001) var max_exploration_rate = 1.0
export(float, 0.0, 1.0, 0.001) var min_exploration_rate = 0.0
export(float) var exploration_rate_decay_time = 0.0
export(bool) var experience_replay = false
export(int) var experience_pool_size = 40
export(float) var think_time = 0.1

var already_hit = []
var velocity = Vector2()
var action = ActionClass.compose(ActionClass.IDLE, ActionClass.DOWN)
var can_act = true
var controller
var controller_name
var network_id = null
var knockback = Vector2()
var invulnerable = false

onready var life = self.max_life
onready var character_type = self.get_script().get_path().get_file().get_basename()
onready var anim_node = $Sprite/AnimationPlayer
onready var Action = ActionClass.new()
onready var debug_mode = global.find_entity("main").character_debug

func _init_ai_controller(params):
	var AIControllerScript = self._get_ai_controller_script()
	self.controller.set_script(AIControllerScript)
	self.add_child(self.controller)
	var init_params = {
		"ai_type": self.ai_type,
		"learning_activated": self.learning_activated,
		"learning_rate": self.learning_rate,
		"discount": self.discount,
		"max_exploration_rate": self.max_exploration_rate,
		"min_exploration_rate": self.min_exploration_rate,
		"exploration_rate_decay_time": self.exploration_rate_decay_time,
		"experience_replay": self.experience_replay,
		"experience_pool_size": self.experience_pool_size,
		"think_time": self.think_time,
		"character_type": self.character_type,
		"network_id": self.network_id,
		"debug_mode": self.debug_mode
	}
	for key in init_params.keys():
		if params.has(key):
			init_params[key] = params[key]
	self.controller.init(init_params)
	if self.debug_mode:
		$Sprite.modulate = self.controller.color

func _get_ai_controller_script():
	var filepath = self.get_script().get_path().get_basename()
	return load(filepath + "RobotController.gd")

func _ready():
	self.anim_node.play(Action.to_string(self.action))
	self.controller = ControllerNode.instance()
	self.set_max_life(self.max_life)
	$LifeBar.value = self.life
	match self.controller_type:
		Controller.PLAYER:
			self.controller.set_script(PlayerController)
			self.add_child(self.controller)
			self.controller_name = "Player"
			self.add_to_group("player")
		Controller.AI:
			self.controller_name = ai_name[self.ai_type]
			self.add_to_group("robot")

func init(params):
	self.network_id = global.dict_get(params, "network_id", null)
	if params.has("damage"):
		# Assert damage is positive
		assert(params.damage >= 0)
		self.damage = params.damage
	if params.has("defense"):
		# Assert defense is positive
		assert(params.defense >= 0)
		self.defense = params.defense
	if params.has("speed"):
		# Assert speed is positive and non-zero
		assert(params.speed > 0)
		self.speed = params.speed
	if params.has("ai_type"):
		# Assert AI type exists
		assert(params.ai_type < AIType.size() and params.ai_type >= 0)
		self.ai_type = params.ai_type
	if params.has("max_life"):
		# Assert that character will have life
		assert(params.max_life > 0)
		self.set_max_life(params.max_life)
	if params.has("life") and params.life >= 0:
		self.set_life(min(self.max_life, max(0, params.life)))
	else:
		self.set_life(self.max_life)
	if self.controller_type == Controller.AI:
		self._init_ai_controller(params)

func end():
	self.controller.end()

func _process(delta):
	self._process_action(self.action)

func _physics_process(delta):
	if self.knockback:
		self.move_and_slide(self.knockback)
		self.knockback *= KNOCKBACK_DECAY_RATE
		if self.knockback < KNOCKBACK_MIN:
			self.knockback = Vector2()
	else:
		self.move_and_slide(self.speed * self.velocity)

func is_ai():
	return self.controller_type == Controller.AI

func get_pretty_name():
	return self.name + " (" + self.controller_name + ")"

func set_max_life(new_max_life):
	self.max_life = new_max_life
	$LifeBar.max_value = new_max_life
	# - [max life] * ([FG margin] + 2) / [rect width]
	$LifeBar.min_value = -new_max_life * 8 / 40

func get_max_life():
	return self.max_life

func get_damage():
	return self.damage

func get_defense():
	return self.defense

func set_life(new_life):
	self.life = min(self.max_life, max(0, new_life))
	$LifeBar.value = self.life
	if self.life == 0:
		self.set_action(Action.DEATH)
		$CollisionPolygon2D.disabled = true

func add_life(amount):
	self.set_life(self.life + amount)

func take_damage(damage, knockback=false):
	self.set_life(self.life - max(0.0, damage - self.get_defense()))
	$Sprite.material.set_shader_param("active", true)
	$DamageBlinkTimer.start()
	self.invulnerable = true

func set_movement(new_movement, force=false):
	if (self.action != Action.DEATH and self.can_act or force) and new_movement != Action.get_movement(self.action):
		self.action = Action.compose(new_movement, self.action)
		self.anim_node.play(Action.to_string(self.action))

func set_action(new_action, force=false):
	if (self.action != Action.DEATH and self.can_act or force) and new_action != self.action:
		self.action = new_action
		self.anim_node.play(Action.to_string(self.action))

func attack():
	self.set_movement(Action.ATTACK)

func is_process_action(action):
	return Action.get_movement(action) == Action.IDLE or \
		   Action.get_movement(action) == Action.WALK

func die():
	self.end()
	self.emit_signal("character_death")

func get_knocked_back(other):
	var dir = (self.position - other.position).normalized()
	self.knockback = dir * KNOCKBACK_INITIAL_SPEED / self.weight

func block_action():
	self.can_act = false

func unblock_action():
	self.can_act = true

func before_reset(timeout):
	self.controller.before_reset(timeout)
	
func reset(timeout):
	self.set_life(self.max_life)
	self.set_action(Action.compose(Action.IDLE, Action.DOWN), true)
	$CollisionPolygon2D.disabled = false
	self.controller.reset(timeout)
	
func after_reset(timeout):
	self.controller.after_reset(timeout)

func _is_entity_attackable(entity):
	return entity.is_in_group("damageble") and not entity.invulnerable and \
		   not entity.is_in_group(global.get_team(self)) and not (entity in self.already_hit) and \
		   Action.get_movement(entity.action) != Action.DEATH

func _on_AnimationPlayer_animation_finished(anim_name):
	var death = Action.to_string(Action.DEATH)
	if anim_name.begins_with(death):
		self.die()

func _on_AttackArea_area_entered(area):
	var entity = area.get_parent()
	if self._is_entity_attackable(entity) and \
	   $AttackArea/AttackAreaPolygon.polygon.size() > 1 and \
	   (entity.position - self.position).dot(Action.to_vec(self.action)) >= 0:
		entity.take_damage(self.get_damage())
		entity.get_knocked_back(self)
		self.already_hit.append(entity)

func _on_DamageBlinkTimer_timeout():
	$Sprite.material.set_shader_param("active", false)
	self.invulnerable = false
