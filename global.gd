extends Node

const SAVE_PATH = "user://save.save"

func has_entity(entity_name):
	return len(get_tree().get_nodes_in_group(entity_name)) != 0

func find_entity(entity_name):
	if not has_entity(entity_name):
		return null
	
	var entity_list = get_tree().get_nodes_in_group(entity_name)
	return entity_list[0]

func get_team(entity):
	if entity.is_in_group("team1"):
		return "team1"
	elif entity.is_in_group("team2"):
		return "team2"
	return ""

func get_enemy(entity):
	var enemy_team = "team1"
	if entity.is_in_group("team1"):
		enemy_team = "team2"
	var enemies = get_tree().get_nodes_in_group(enemy_team)
	if len(enemies) == 0:
		return null
	var closest_enemy = null
	var min_dist = INF
	for enemy in enemies:
		var dist = enemy.position.distance_to(entity.position)
		if dist < min_dist:
			closest_enemy = enemy
			min_dist = dist
	return closest_enemy

# Random integer in the interval [start, end] (including both ends)
func randi_range(start, end):
	return int(start + floor(randf() * (end - start + 1)))

func sample_from_normal(mean, std):
	var U = randf()
	var V
	for i in range(5):
		V = randf()
	var X = sqrt(-2 * log(U)) * cos(2 * PI * V)
	return mean + std * X

func sample_from_normal_limited(mean, std, limits=[-INF, INF]):
	var X = self.sample_from_normal(mean, std)
	while X <= limits[0] or X >= limits[1]:
		X = self.sample_from_normal(mean, std)
	return X

func choose_one(array, include_null=false, a=-1, b=-1):
	if a == -1:
		a = 0
	if b == -1:
		b = array.size() - 1
	var size = b - a + int(include_null)
	var rand_num = global.randi_range(a, a + size)
	if include_null and rand_num == a + size:
		return null
	return array[rand_num]

func sample(array, num, include_null=false, a=-1, b=-1):
	if a == -1:
		a = 0
	if b == -1:
		b = array.size()
	var ret = []
	for i in range(num):
		ret.append(self.choose_one(array, include_null, a, b))
	return ret

func choose(array, num):
	if num == len(array):
		return array.copy()
	var ret = []
	var used = {}
	while len(ret) != num:
		var rand_num = self.randi_range(0, len(array) - 1)
		if not used.has(rand_num):
			used[rand_num] = true
			ret.append(array[rand_num])
	return ret

func shuffle_array(array):
	var n = len(array)
	for i in range(n - 1):
		var pos = self.randi_range(i, n - 1)
		var tmp = array[pos]
		array[pos] = array[i]
		array[i] = tmp
	return array

func dict_get(dict, key, default):
	if key == null:
		return null
	if not dict.has(key):
		return default
	return dict[key]

func create_array(size, fill=null):
	var array = []
	for i in range(size):
		array.append(fill)
	return array

func create_matrix(height, width, fill=null):
	var matrix = []
	for i in range(height):
		matrix.append([])
		for j in range(width):
			matrix[i].append(fill)
	return matrix

func max(array):
	if array.size() == 0:
		return null
	var mx = array[0]
	for el in array:
		mx = max(el, mx)
	return mx

func min(array):
	if array.size() == 0:
		return null
	var mn = array[0]
	for el in array:
		mn = min(el, mn)
	return mn

func argmax(array):
	if array.size() == 0:
		return null
	var mx = array[0]
	var idx = 0
	for i in range(1, array.size()):
		if mx < array[i]:
			mx = array[i]
			idx = i
	return idx

func save_info(nodes):
	var saveFile = File.new()
	var saveInfo = []
	if saveFile.file_exists(SAVE_PATH):
		saveFile.open(SAVE_PATH, saveFile.READ)
		saveInfo = JSON.parse(saveFile.get_as_text()).result
		saveFile.close()
	else:
		saveInfo = []

	var newInfos = {}
	for node in nodes:
		var name = node.get_name()
		var info = node.get_info()
		newInfos[name] = info
	
	saveInfo.append(newInfos)

	saveFile.open(SAVE_PATH, saveFile.WRITE)
	saveFile.store_string(JSON.print(saveInfo))
	saveFile.close()

func sum(array):
	var sum = 0.0
	for el in array:
		sum += el
	return sum

func norm(array):
	var sum = 0.0
	for el in array:
		sum += el * el
	return sqrt(sum)

func normalize(array):
	var norm = self.norm(array)
	for i in range(array.size()):
		array[i] /= norm