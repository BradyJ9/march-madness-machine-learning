from backprop import *
from BracketSimulation import *
import numpy as np
import arff
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

year_and_game_data_rows = {
  2021: (2,133),
  2019: (134,261),
  2018: (262,395),
  2017: (396,529),
  2016: (530,663),
  2015: (664,797),
  2014: (798,931),
  2013: (932,1065),
  2012: (1066,1199),
  2011: (1200,1331),
  2010: (1332,1459),
}

class Tests():
  
  #variables:
    #hidden layer nodes - start low, double until 
    #output nodes...1
    #momentum
    #lr
    #shuffle data
    #stopping criteria - lack of test improvement OR validation set
                        #we don't want any of the test data to be seen or trained on, so the 2 brackets we will test with should be 
                        #part of the "test data" while we should create separate validation data for stopping criteria
    #normalizing data - probably should??
    #epochs - non-deterministic...will vary
    #initial weights - random
    #epochs_since_best_val_set
    
  #data to graph: (based on 3 years)
    #average values out of 10 runs (no graph, just metrics)
    #total correct vs some hyperparameter (convergence criteria)
    #graph with total score vs average human
    
  #process:
    #train, test, create bracket
    
  
  def tourney_data_train(self):
    hidden_layer_widths = [2]
    mlp_debug = MLP(lr=0.5, momentum=0.9, shuffle=True, hidden_layer_widths=hidden_layer_widths)
        
    print("\nBrady Bess\n***TOURNEY DATA - 1 OUTPUT NODE***\n")
    d = arff.load(open('./AdjustedData.arff', 'r'))
    data = d['data']
    data, col_ranges = self.normalize_data(data)
    
    train_data, test_data = self.split_train_test(data, [2011,2020])
    train_data, validation_data = self.split_train_validate(train_data)
    
    train_wb = mlp_debug.add_bias(train_data)
    train_X = train_wb[:, :-1]
    train_y = train_wb[:, -1]
    
    validation_wb = mlp_debug.add_bias(validation_data)
    validation_X = validation_wb[:, :-1]
    validation_y = validation_wb[:, -1]
    
    test_wb = mlp_debug.add_bias(test_data)
    test_X = test_wb[:, :-1]
    test_y = test_wb[:, -1]
    
    mlp_debug.init_network(len(train_X[0]) - 1, len(hidden_layer_widths), hidden_layer_widths[0], 1) #-1 because bias is added already
    mlp_debug.network.to_string()
    
    mlp_debug.fit(train_X, train_y, validation_X=validation_X, validation_y=validation_y, use_validation_set=True)
    
    #print("\n\nFinal Network\n")
    mlp_debug.network.to_string()
    
    weights = mlp_debug.get_weights() 
    #print("\n\nFinal Weights:\n")
    #print(weights)
    
    bracket_2021 = BracketSimulation(mlp_debug, 2011, col_ranges)
    
    bracket_2021.simulate_tourney()
    
    bracket_2021.score_simulated_bracket()
    
    bracket_2011 = BracketSimulation(mlp_debug, 2020, col_ranges)
    
    bracket_2011.simulate_tourney()
    
    bracket_2011.score_simulated_bracket()
    

  def virg_umbc_test(self):
    for i in range(100):
      hidden_layer_widths = [2]
      mlp_debug = MLP(lr=0.5, momentum=0.9, shuffle=True, hidden_layer_widths=hidden_layer_widths)
          
      d = arff.load(open('./AdjustedData.arff', 'r'))
      data = d['data']
      data, col_ranges = self.normalize_data(data)
      
      train_data, test_data = self.split_train_test(data, [2018])
      train_data, validation_data = self.split_train_validate(train_data)
      
      train_wb = mlp_debug.add_bias(train_data)
      train_X = train_wb[:, :-1]
      train_y = train_wb[:, -1]
      
      validation_wb = mlp_debug.add_bias(validation_data)
      validation_X = validation_wb[:, :-1]
      validation_y = validation_wb[:, -1]
      
      test_wb = mlp_debug.add_bias(test_data)
      test_X = test_wb[:, :-1]
      test_y = test_wb[:, -1]
      
      mlp_debug.init_network(len(train_X[0]) - 1, len(hidden_layer_widths), hidden_layer_widths[0], 1) #-1 because bias is added already
      #mlp_debug.network.to_string()
      
      mlp_debug.fit(train_X, train_y, validation_X=validation_X, validation_y=validation_y, use_validation_set=True)
      
      #print("\n\nFinal Network\n")
      #mlp_debug.network.to_string()
      
      weights = mlp_debug.get_weights() 
      #print("\n\nFinal Weights:\n")
      #print(weights)
      
      #test to see if mlp can predict first 16-1 upset in 2018
      virg_umbc = BracketSimulation(mlp_debug, 2018, col_ranges)
      virg_umbc.simulate_tourney()


  ###UTIL FUNCTIONS###
  
  def split_train_test(self, data, years=[]):
    
    if len(years) == 0:
        test_perc=.2
        test_data = []
        num_in_test = round(len(data) * test_perc)

        random.shuffle(data)

        for i in range(num_in_test):
          test_el = data.pop()
          test_data.append(test_el)

        #print("Training DATA: " + str(len(data)) + " elements...")
        #print(data) #leftver training data
        #print("\n\nTEST DATA: " + str(len(test_data)) + " elements...")
        #print(test_data) #test data
        
        return data, test_data
    else:
      test_data = []
      train_data = []
      for year in year_and_game_data_rows.keys():
        year_rows = year_and_game_data_rows[year]
        is_test_year = False
        for y in years:
          if y == year:
            is_test_year = True
            test_data.extend(data[ year_rows[0] - 2 : year_rows[1] - 1 ])
        
        if not is_test_year:
          train_data.extend(data[ year_rows[0] - 2 : year_rows[1] - 1 ])
        
      return train_data, test_data
            
  def split_train_validate(self, data):
      val_perc=.15
      val_data = []
      num_in_test = round(len(data) * val_perc)

      random.shuffle(data)

      for i in range(num_in_test):
        val_el = data.pop()
        val_data.append(val_el)
      
      return data, val_data
      
  def normalize_data(self, data_X):
    col_ranges = {}
    for c in range(len(data_X[0])): #how many columns need normalized
      try:
        x = float(data_X[0][c])
      except ValueError:
        print("ERROR NORM DATA")
      col_max, col_min = -1, 1000000
      for r in range(len(data_X)): #find xmin and xmax
        if data_X[r][c] is None:
          print("ERROR NORM DATA")
        if float(data_X[r][c]) > col_max:
          col_max = float(data_X[r][c])
        if float(data_X[r][c]) < col_min:
          col_min = float(data_X[r][c])
      
      col_ranges[c] = (col_min, col_max)
      
      for r in range(len(data_X)):
        if not data_X[r][c]: #it is counting 0 as true, ignore this
          #print("ERROR NORM DATA")
          pass
        data_X[r][c] = float((float(data_X[r][c]) - col_min) / (col_max - col_min))
    return data_X, col_ranges
    
        

t = Tests()

t.tourney_data_train()
#t.virg_umbc_test()

