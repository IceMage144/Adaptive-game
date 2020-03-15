extends "res://Databases/Monsters/MonsterDataBase.gd"

enum AIType { PERCEPTRON, SINGLE, MEMORY, MULTI }

const ATTRIBUTES = ["ai_type", "learning_activated", "learning_rate",
                    "discount", "max_exploration_rate", "min_exploration_rate",
                    "exploration_rate_decay_time", "experience_replay",
                    "experience_pool_size", "think_time"]

export(AIType) var ai_type = AIType.PERCEPTRON
export(bool) var learning_activated = true
export(float, 1.0, 0.0, 0.001) var learning_rate = 0.0
export(float, 0.0, 1.0, 0.001) var discount = 0.0
export(float, 0.0, 1.0, 0.001) var max_exploration_rate = 1.0
export(float, 0.0, 1.0, 0.001) var min_exploration_rate = 0.0
export(float) var exploration_rate_decay_time = 0.0
export(bool) var experience_replay = false
export(int) var experience_pool_size = 40
export(float) var think_time = 0.1