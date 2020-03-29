extends Object

const type2str = {
	TYPE_NIL: "nil",
	TYPE_BOOL: "bool",
	TYPE_INT: "int",
	TYPE_REAL: "float",
	TYPE_STRING: "String",
	TYPE_VECTOR2: "Vector2",
	TYPE_RECT2: "Rect2",
	TYPE_VECTOR3: "Vector3",
	TYPE_TRANSFORM2D: "Transform2D",
	TYPE_PLANE: "Plane",
	TYPE_QUAT: "Quat",
	TYPE_AABB: "AABB",
	TYPE_BASIS: "Basis",
	TYPE_TRANSFORM: "Transform",
	TYPE_COLOR: "Color",
	TYPE_NODE_PATH: "NodePath",
	TYPE_RID: "RID",
	TYPE_OBJECT: "Object",
	TYPE_DICTIONARY: "Dictionary",
	TYPE_ARRAY: "Array",
	TYPE_RAW_ARRAY: "PoolByteArray",
	TYPE_INT_ARRAY: "PoolIntArray",
	TYPE_REAL_ARRAY: "PoolRealArray",
	TYPE_STRING_ARRAY: "PoolStringArray",
	TYPE_VECTOR2_ARRAY: "PoolVector2Array",
	TYPE_VECTOR3_ARRAY: "PoolVector3Array",
	TYPE_COLOR_ARRAY: "PoolColorArray"
}

const NO_SIZE = -1

class CompositeType:
	var types
	var size

	func _init(types, size = NO_SIZE):
		if typeof(types) == TYPE_INT:
			types = [types]
		self.types = types
		self.size = size
	
	func is_type_of(obj):
		var res = false
		for type in self.types:
			res = res or typeof(obj) == type
		return res and (self.size == NO_SIZE or obj.size() == self.size)
	
	func _str():
		var type_str
		if self.types.size() == 1:
			type_str = type2str[self.types[0]]
		else:
			type_str = "{"
			for type in self.types:
				type_str += type2str[type] + ","
			type_str = type_str.substr(0, type_str.length() - 1) + "}"
		if self.size == NO_SIZE:
			return type_str
		return type_str + "[" + str(self.size) + "]"

func print_error(msg):
	printerr("Error: ", msg)
	print_stack()

func assert(cond, msg):
	if not cond:
		self.print_error(msg)
	return cond

func str_type(obj):
	return type2str[typeof(obj)]

func is_compose_type(obj, compose_type):
	if not OS.is_debug_build():
		return true
	return self._is_compose_type_r(obj, compose_type, 0)

func _is_compose_type_r(obj, compose_type, depth):
	if not compose_type[depth].is_type_of(obj):
		return false
	if typeof(obj) == TYPE_ARRAY:
		for i in range(obj.size()):
			if not self._is_compose_type_r(obj[i], compose_type, depth + 1):
				return false
	return true

