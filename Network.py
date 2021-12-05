

class Network():
  
  BIAS = 1
  INPUT_LAYER = 0
  HIDDEN_LAYER = 1
  OUTPUT_LAYER = 2
  
  def __init__(self, input_nodes, hidden_layers, hidden_layer_width, output_nodes):
    
    
    self.layers = [] #assuming 3, only one hidden layer
    self.input_layer_nodes = input_nodes + self.BIAS
    self.hidden_layer_nodes = hidden_layer_width + self.BIAS #assuming only one hidden layer
    self.output_layer_nodes = output_nodes

    input_layer = Layer(-1, input_nodes + self.BIAS, None)
    
    #hidden_layers = []
    #for l in range(len(hidden_layers)):
    #  hl = Layer(0, hidden_layer_width + BIAS, output_nodes if l == (len(hidden_layers) - 1) else hidden_layer_width)
    #  hidden_layers.append(hl)
    #
    #For now we are assuming there is only one hidden layer
    
    hidden_layer = Layer(0, hidden_layer_width + self.BIAS, input_nodes + self.BIAS)
    
    output_layer = Layer(1, output_nodes, hidden_layer_width + self.BIAS)
    
    #INSERT LAYERS#
    self.layers.append(input_layer)
    #for i in range(len(hidden_layers)):
    #  self.layers.append(hidden_layers[i])
    self.layers.append(hidden_layer)
    self.layers.append(output_layer)
    
  def init_weights(self, weights):
    if len(weights) != self.hidden_layer_nodes - self.BIAS + self.output_layer_nodes: #-1 for hidden layer bias
      raise Exception("Incorrect weight config...too many weight arrays")
    
    for h in range(self.hidden_layer_nodes - self.BIAS): #init hidden layer nodes weights coming in
      if len(self.layers[self.HIDDEN_LAYER].connected_weights[h]) != len(weights[h]):
        raise Exception("Incorrect weight config...mismatched sizes of weight arrays")
      self.layers[self.HIDDEN_LAYER].connected_weights[h] = weights[h] #hidden layer
      
    for o in range(self.hidden_layer_nodes - self.BIAS, self.hidden_layer_nodes - self.BIAS + self.output_layer_nodes):
      if len(self.layers[self.OUTPUT_LAYER].connected_weights[o - self.hidden_layer_nodes + self.BIAS]) != len(weights[o]):
        raise Exception("Incorrect weight config...mismatched sizes of weight arrays")
      self.layers[self.OUTPUT_LAYER].connected_weights[o - self.hidden_layer_nodes + self.BIAS] = weights[o] #hidden layer
      
    #debug
    #print("Done INIT Weights in Network...")
    #self.to_string()
    
  def get_weights(self):
    w = []
    for l in range(1, len(self.layers)): #1 to exclude input layer w no weights
      w_p = self.layers[l].get_weights()
      for wt in range(len(w_p)):
        w.append(w_p[wt])
    return w      
    
  def to_string(self):
    print("MLP NETWORK")
    print("Layers: " + str(len(self.layers)))
    for x in range(len(self.layers)):
      self.layers[x].to_string()
    
class Layer():
  
  def __init__(self, layer_type, nodes, bwd_connected_nodes):
    
    self.layer_type = layer_type #-1 input layer, 0 hidden layer, 1 output layer
    self.nodes = [Node() for n in range(nodes)]
    
    nodes_w_weights = len(self.nodes) if self.layer_type == 1 else len(self.nodes) - 1
    self.connected_weights = [[None for y in range(bwd_connected_nodes)] for x in range(nodes_w_weights)] if bwd_connected_nodes else None
    #backward connections ^^^
    
  def init_weights(self, weights : [[]], layer):
    self.connected_weights = weights
    pass
  
  def to_string(self):
    title = ""
    if self.layer_type == -1:
      title = "\tINPUT"
    elif self.layer_type == 0:
      title = "\tHIDDEN"
    elif self.layer_type == 1:
      title = "\tOUTPUT"
      
    print(title + " LAYER: " + str(len(self.nodes)) + " nodes")
    print("\t\tWEIGHTS COMING INTO LAYER:")
    if self.connected_weights:
      for x in range(len(self.connected_weights)):
        print("\t\t\t" + str(self.connected_weights[x]))
        
  def get_weights(self):
    if self.layer_type != -1: #input layer has no weights
      w = []
      nodes_w_weights_incoming = len(self.nodes) - 1 if self.layer_type == 0 else len(self.nodes)
      for n in range(nodes_w_weights_incoming): # -1 for bias
        w.append(self.connected_weights[n])
      return w
    else:
      return None
      
        
class Node():
  
  def __init__(self):
    
    self.net = 0
    self.z_act_fun = 0
    self.error = 0

