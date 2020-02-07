extends Object

var table = {}

func push(name, val):
	if not self.table.has(name):
		self.table[name] = []
	self.table[name].append(val)

func avg(name):
	if not self.table.has(name):
		return 0.0
	return global.sum(self.table[name]) / self.table[name].size()

func max(name):
	if not self.table.has(name):
		return 0.0
	return global.max(self.table[name])

func min(name):
	if not self.table.has(name):
		return 0.0
	return global.min(self.table[name])

func sum(name):
	if not self.table.has(name):
		return 0.0
	return global.sum(self.table[name])

func flush(name):
	self.table[name] = []

func size(name):
	return self.table[name].size()

func print_stats(name, stats_list):
	if not self.table.has(name):
		return
	print(name + " (" + str(self.size(name)) + " values):")
	for stat in stats_list:
		if self.has_method(stat):
			print("\t" + stat + ": " + str(callv(stat, [name])))

func get_stored(name):
	if not self.table.has(name):
		return []
	return self.table[name]