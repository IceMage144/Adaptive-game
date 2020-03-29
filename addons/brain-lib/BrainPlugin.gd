tool
extends EditorPlugin

func _enter_tree():
	add_custom_type("NeuralNetwork1DAdam", "Node", preload("NeuralNetwork1DAdam.gd"), preload("icons/nn1d_adam.png"))
	add_custom_type("NeuralNetwork1DAdagrad", "Node", preload("NeuralNetwork1DAdagrad.gd"), preload("icons/nn1d_adagrad.png"))
	add_custom_type("NeuralNetwork1DAdamax", "Node", preload("NeuralNetwork1DAdamax.gd"), preload("icons/nn1d_adamax.png"))
	add_custom_type("NeuralNetwork1DGradientDescent", "Node", preload("NeuralNetwork1DGradientDescent.gd"), preload("icons/nn1d_gd.png"))
	add_custom_type("NeuralNetwork1DMomentum", "Node", preload("NeuralNetwork1DMomentum.gd"), preload("icons/nn1d_momentum.png"))
	add_custom_type("NeuralNetwork1DNesterovMomentum", "Node", preload("NeuralNetwork1DNesterovMomentum.gd"), preload("icons/nn1d_nesterov.png"))
	add_custom_type("NeuralNetwork1DRMSprop", "Node", preload("NeuralNetwork1DRMSprop.gd"), preload("icons/nn1d_rmsprop.png"))
	add_custom_type("FullyConnected", "Node", preload("FullyConnected.gd"), preload("icons/fc.png"))
	add_custom_type("Dropout", "Node", preload("Dropout.gd"), preload("icons/dropout.png"))
	add_custom_type("Tanh", "Node", preload("Tanh.gd"), preload("icons/tanh.png"))
	add_custom_type("Sigmoid", "Node", preload("Sigmoid.gd"), preload("icons/sigmoid.png"))
	add_custom_type("ASinH", "Node", preload("ASinH.gd"), preload("icons/asinh.png"))
	add_custom_type("ReLU", "Node", preload("ReLU.gd"), preload("icons/relu.png"))
	add_custom_type("LSTM", "Node", preload("LSTM.gd"), preload("icons/lstm.png"))

func _exit_tree():
	remove_custom_type("NeuralNetwork1DAdam")
	remove_custom_type("NeuralNetwork1DAdagrad")
	remove_custom_type("NeuralNetwork1DAdamax")
	remove_custom_type("NeuralNetwork1DGradientDescent")
	remove_custom_type("NeuralNetwork1DMomentum")
	remove_custom_type("NeuralNetwork1DNesterovMomentum")
	remove_custom_type("NeuralNetwork1DRMSprop")
	remove_custom_type("FullyConnected")
	remove_custom_type("Dropout")
	remove_custom_type("Tanh")
	remove_custom_type("ASinH")
	remove_custom_type("Sigmoid")
	remove_custom_type("ReLU")
	remove_custom_type("LSTM")