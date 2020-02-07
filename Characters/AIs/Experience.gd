extends Object

const MINIMUM_SIZE_FACTOR = 3
const NO_DEPTH = 0

var sample_size
var max_depth

var depth = 0
var keys = []
var features = []
var rewards = []
var next_states = []
var actions = []

# int, int -> void
func _init(sample_size, max_depth = NO_DEPTH):
    self.sample_size = sample_size
    self.max_depth = max_depth
    if self._has_depth():
        self._push_clean()

# -> bool
func _has_depth():
    return self.max_depth != NO_DEPTH

# -> bool
func _is_full():
    return not self._has_depth() or self.depth == self.max_depth

# -> void
func _push_clean():
    self.keys.append(self.keys.size())
    self.features.append([])
    self.rewards.append([])
    self.next_states.append([])
    self.actions.append([])
    self.depth = 0

# Features, Reward, State, Action -> void
func push(feat, reward, next_state, action = null):
    if self._has_depth():
        self.features.back().append(feat)
        self.rewards.back().append(reward)
        self.next_states.back().append(next_state)
        self.actions.back().append(action)
        self.depth += 1
        if self._is_full():
            self._push_clean()
    else:
        self.keys.append(self.keys.size())
        self.features.append(feat)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.actions.append(action)

# -> void
func pop():
    self.keys.pop_back()
    self.features.pop_back()
    self.rewards.pop_back()
    self.next_states.pop_back()
    self.actions.pop_back()
    self.depth = 0
    if self._has_depth():
        self._push_clean()

# -> void
func clean_end():
    if self.depth > 0:
        self.pop()

# -> bool
func end_is_clean():
    return self.depth == 0

# (max_depth == 0) -> Array{ Array[Features], Array[Reward], Array[State], Array[Action] }
# (max_depth != 0) -> Array{ Array[Array[Features]], Array[Array[Reward]], Array[Array[State]], Array[Array[Action]] }
func sample():
    var ret = [[], [], [], []]
    if self.keys.size() > MINIMUM_SIZE_FACTOR * self.sample_size:
        var sample_keys = global.sample(self.keys, self.sample_size, false, 0, self.keys.size() - 2)
        for key in sample_keys:
            ret[0].append(self.features[key])
            ret[1].append(self.rewards[key])
            ret[2].append(self.next_states[key])
            ret[3].append(self.actions[key])
    return ret

# (max_depth == 0) -> Array{ Array[Features], Array[Reward], Array[State], Array[Action] }
# (max_depth != 0) -> Array{ Array[Array[Features]], Array[Array[Reward]], Array[Array[State]], Array[Array[Action]] }
func simple_sample():
    var ret = [[], [], [], []]
    var sample_key = global.choose_one(self.keys, false, 0, self.keys.size() - 2)
    ret[0].append(self.features[sample_key])
    ret[1].append(self.rewards[sample_key])
    ret[2].append(self.next_states[sample_key])
    ret[3].append(self.actions[sample_key])
    return ret