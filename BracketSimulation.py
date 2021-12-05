import numpy as np
import csv

years_and_first_round_matchups = {
  2021: ["Gonzaga", "Norfolk State", "Oklahoma", "Missouri", "Creighton", "UC-Santa Barbara", "Virginia", "Ohio", "Southern California", "Drake", "Kansas", "Eastern Washington",
         "Oregon", "Virginia Commonwealth", "Iowa", "Grand Canyon", "Michigan", "Texas Southern", "Louisiana State", "St. Bonaventure", "Colorado", "Georgetown", "Florida State", "North Carolina-Greensboro",
         "Brigham Young", "UCLA", "Texas", "Abilene Christian", "Connecticut", "Maryland", "Alabama", "Iona", "Baylor", "Hartford", "North Carolina", "Wisconsin", "Villanova", "Winthrop", "Purdue", "North Texas",
         "Texas Tech", "Utah State", "Arkansas", "Colgate", "Florida", "Virginia Tech", "Ohio State", "Oral Roberts", "Illinois", "Drexel", "Loyola (IL)", "Georgia Tech", "Tennessee", "Oregon State", "Oklahoma State", "Liberty",
         "San Diego State", "Syracuse", "West Virginia", "Morehead State", "Clemson", "Rutgers", "Houston", "Cleveland State"],
  2016: ["Kansas", "Austin Peay", "Colorado", "Connecticut","Maryland","South Dakota State","University of California","Hawaii","Arizona","Wichita State","Miami (FL)","Buffalo","Iowa","Temple","Villanova","North Carolina-Asheville","Oregon","Holy Cross","Saint Joseph's","Cincinnati","Baylor","Yale","Duke","North Carolina-Wilmington","Texas","Northern Iowa",
         "Texas A&M","Green Bay","Oregon State","Virginia Commonwealth","Oklahoma","Cal State Bakersfield","North Carolina","Florida Gulf Coast","Southern California","Providence","Indiana","Chattanooga","Kentucky","Stony Brook","Notre Dame","Michigan","West Virginia","Stephen F. Austin","Wisconsin","Pittsburgh","Xavier","Weber State","Virginia","Hampton","Texas Tech","Butler","Purdue","Little Rock",
         "Iowa State","Iona","Seton Hall","Gonzaga","Utah","Fresno State","Dayton","Syracuse","Michigan State","Middle Tennessee"],
  
  2014: ["Florida","Albany (NY)","Colorado","Pittsburgh","Virginia Commonwealth","Stephen F. Austin","UCLA","Tulsa","Ohio State","Dayton","Syracuse","Western Michigan","New Mexico","Stanford","Kansas","Eastern Kentucky","Virginia","Coastal Carolina","Memphis","George Washington","Cincinnati","Harvard","Michigan State","Delaware","North Carolina","Providence","Iowa State","North Carolina Central","Connecticut","Saint Joseph's",
         "Villanova","Milwaukee","Arizona","Weber State","Gonzaga","Oklahoma State","Oklahoma","North Dakota State","San Diego State","New Mexico State","Baylor","Nebraska","Creighton","Louisiana","Oregon","Brigham Young","Wisconsin","American","Wichita State","Cal Poly","Kentucky","Kansas State","Saint Louis","North Carolina State","Louisville","Manhattan","Massachusetts","Tennessee","Duke","Mercer","Texas","Arizona State","Michigan","Wofford"],
  
  2011: ["Ohio State", "Texas-San Antonio", "George Mason","Villanova","West Virginia","Clemson","Kentucky","Princeton","Xavier","Marquette","Syracuse","Indiana State","Washington","Georgia","North Carolina","Long Island University","Duke","Hampton","Michigan","Tennessee","Arizona","Memphis",
         "Texas","Oakland","Cincinnati","Missouri","Connecticut","Bucknell","Temple","Penn State","San Diego State", "Northern Colorado","Kansas","Boston University","Nevada-Las Vegas","Illinois","Vanderbilt","Richmond","Louisville","Morehead State","Georgetown","Virginia Commonwealth","Purdue","Saint Peter's",
         "Texas A&M","Florida State","Notre Dame","Akron","Pittsburgh","North Carolina-Asheville","Butler","Old Dominion","Kansas State","Utah State","Wisconsin","Belmont","St. John's (NY)","Gonzaga","Brigham Young","Wofford","UCLA","Michigan State","Florida","UC-Santa Barbara"],
  
  2018: ["Virginia","Maryland-Baltimore County"]
  }

years_and_correct_games_for_rounds = {
  2021: [["Gonzaga","Oklahoma","Creighton","Ohio","Southern California","Kansas","Oregon","Iowa","Michigan","Louisiana State","Colorado","Florida State","UCLA","Abilene Christian","Maryland","Alabama", "Baylor","Wisconsin","Villanova","North Texas","Texas Tech","Arkansas","Florida","Oral Roberts","Illinois","Loyola (IL)","Oregon State","Oklahoma State","Syracuse","West Virginia","Rutgers","Houston"],
         ["Gonzaga","Creighton","Southern California","Oregon","Michigan","Florida State","UCLA","Alabama","Baylor","Villanova","Arkansas","Oral Roberts","Loyola (IL)","Oregon State","Syracuse","Houston"],
         ["Gonzaga","Southern California","Michigan","UCLA","Baylor","Arkansas","Oregon State","Houston"],
         ["Gonzaga","UCLA","Baylor","Houston"],
         ["Gonzaga","Baylor"],
         ["Baylor"]],
  2016: [["Kansas","Connecticut","Maryland","Hawaii","Wichita State","Miami (FL)","Iowa","Villanova","Oregon","Saint Joseph's","Yale","Duke","Northern Iowa","Texas A&M","Virginia Commonwealth","Oklahoma","North Carolina","Providence","Indiana","Kentucky","Notre Dame", "Stephen F. Austin","Wisconsin","Xavier","Virginia","Butler","Little Rock","Iowa State","Gonzaga","Utah","Syracuse","Middle Tennessee"],
         ["Kansas","Maryland","Miami (FL)","Villanova","Oregon","Duke","Texas A&M","Oklahoma","North Carolina","Indiana","Notre Dame","Wisconsin","Virginia","Iowa State","Gonzaga","Syracuse"],
         ["Kansas","Villanova","Oregon","Oklahoma","North Carolina","Notre Dame","Virginia","Syracuse"],
         ["Villanova","Oklahoma","North Carolina","Syracuse"],
         ["Villanova","North Carolina"],
         ["Villanova"]
         ],
  2014: [["Florida", "Pittsburgh","Stephen F. Austin","UCLA","Dayton","Syracuse","Stanford","Kansas","Virginia","Memphis","Harvard","Michigan State","North Carolina","Iowa State","Connecticut","Villanova","Arizona","Gonzaga","North Dakota State","San Diego State","Baylor","Creighton","Oregon","Wisconsin","Wichita State","Kentucky","Saint Louis","Louisville","Tennessee","Mercer","Texas","Michigan"],
         ["Florida","UCLA","Dayton","Stanford","Virginia","Michigan State","Iowa State","Connecticut","Arizona","San Diego State","Baylor","Wisconsin","Kentucky","Louisville","Tennessee","Michigan"],
         ["Florida","Dayton","Michigan State","Connecticut","Arizona","Wisconsin","Kentucky","Michigan"],
         ["Florida","Connecticut","Wisconsin","Kentucky"],
         ["Connecticut","Kentucky"],
         ["Connecticut"]
        ],
  2011: [["Ohio State","George Mason","West Virginia","Kentucky","Marquette","Syracuse","Washington","North Carolina","Duke","Michigan","Arizona","Texas","Cincinnati","Connecticut","Temple","San Diego State","Kansas","Illinois","Richmond","Morehead State","Virginia Commonwealth","Purdue","Florida State","Notre Dame","Pittsburgh","Butler","Kansas State","Wisconsin","Gonzaga", "Brigham Young","UCLA","Florida"],
         ["Ohio State","Kentucky", "Marquette","North Carolina","Duke","Arizona","Connecticut","San Diego State","Kansas","Richmond","Virgina Commonwealth","Florida State","Butler","Wisconsin","Brigham Young","Florida"],
         ["Kentucky","North Carolina","Arizona","Connecticut","Kansas","Virginia Commonwealth","Butler","Florida"],
         ["Kentucky","Connecticut","Virgina Commonwealth","Butler"],
         ["Connecticut","Butler"],
         ["Connecticut"]],
  
  2018: ["Maryland-Baltimore County"]
  }

class BracketSimulation():
  
  def __init__(self, mlp_model, year, col_ranges):
    self.mlp = mlp_model
    
    self.year = year #2010-2021
    
    self.normalization_values = col_ranges
    
    self.team_stats = self.get_team_stats_for_year_from_csv("TeamStats.csv", str(self.year))

    self.rounds = [Round("First Round", years_and_first_round_matchups[self.year])] #array of Rounds
      
    
  # Reads in a csv file and returns it as a 2D array
  def get_team_stats_for_year_from_csv(self, filename, year):
    rows = []
    with open(filename, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
          if str(year) in str(row) and str('NCAA') in str(row):
            rows.append(row)
    return rows
  
  def read_csv(self, filename):
    rows = []
    with open(filename, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csv_reader:
            rows.append(row)
    return rows
        
  def simulate_tourney(self):
    while len(self.rounds[len(self.rounds) - 1].games) > 1:
      print("\n\n" + self.rounds[len(self.rounds) - 1].round_name + ":\n")
      curr_round = self.rounds[len(self.rounds) - 1]
      next_round = []
      for t in range(int(len(curr_round.games) / 2)):
        team_a = self.find_row_of_team(curr_round.games[t*2])
        team_b = self.find_row_of_team(curr_round.games[t*2+1])
        winner = self.simulate_game(team_a, team_b)
        next_round.append(winner)
      self.rounds.append(Round(str(len(self.rounds[len(self.rounds) - 1].games) / 2) + " teams remaining", next_round))
      
  def find_row_of_team(self, team_name):
    for team in self.team_stats:
      if str(team_name) + " NCAA" == str(team[2]):
        return team
      
  def simulate_game(self, team1, team2):
    stats = self.read_csv('TeamStats.csv')
    stats_header = stats[1]
    stats = stats[2:]
    
    index_school_name = stats_header.index('School')
    # index of column in CSV with stat
    indexes_unnormalized = {
        'W-L%': stats_header.index('W-L%'),
        'SOS': stats_header.index('SOS'),
        'FG%': stats_header.index('FG%'),
        '3P%': stats_header.index('3P%'),
        'FT%': stats_header.index('FT%'),
    }
    indexes_normalized = {
        'PPG': stats_header.index('Tm.'),
        'Allowed PPG': stats_header.index('Opp.'),
        'ORB': stats_header.index('ORB'),
        'Rebounds': stats_header.index('TRB'),
        'Assists': stats_header.index('AST'),
        'Steals': stats_header.index('STL'),
        'Blocks': stats_header.index('BLK'),
        'Turnovers': stats_header.index('TOV'),
        'PF': stats_header.index('PF'),
    }
    index_games_played = stats_header.index('G')

    num_games_1 = float(team1[index_games_played])
    num_games_2 = float(team2[index_games_played])
  
    diff = []
    column = 0
    for curr_stat in indexes_unnormalized:
        index = indexes_unnormalized[curr_stat]
        v_diff = float(team1[index]) - float(team2[index])
        v_diff = self.normalize_col_on_index(v_diff, column)
        diff.append(v_diff)
        column += 1
    for curr_stat in indexes_normalized:
        index = indexes_normalized[curr_stat]
        v_diff = (float(team1[index]) / num_games_1) - (float(team2[index]) / num_games_2)
        v_diff = self.normalize_col_on_index(v_diff, column)
        diff.append(v_diff)
        column += 1

    diff.append(float(1)) #bias input
    result = self.mlp.predict([diff])
    
    if result[0] == 0: #team_a lost
      print(str(team1[index_school_name][:-5]) + " LOSES TO " + str(team2[index_school_name][:-5]))
      return team2[index_school_name][:-5]
    elif result[0] == 1: #team_a won
      print(str(team1[index_school_name][:-5]) + " DEFEATS " + str(team2[index_school_name][:-5]))
      return team1[index_school_name][:-5]
    else:
      raise Exception("Error predicting winner")
      
  def normalize_col_on_index(self, v_diff, index):
    index_range = self.normalization_values[index]
    col_min = index_range[0]
    col_max = index_range[1]
    
    return float((v_diff - col_min) / (col_max - col_min))
      
  def score_simulated_bracket(self):
    scoring_scale = [1,2,4,8,16,32]
    
    points = 0
    correct_guesses = {"32.0 teams remaining":0,
                       "16.0 teams remaining":0,
                       "8.0 teams remaining":0,
                       "4.0 teams remaining":0,
                       "2.0 teams remaining":0,
                       "1.0 teams remaining":0}
    
    for r in range(1, len(self.rounds)):
      if len(self.rounds[r].games) != len(years_and_correct_games_for_rounds[self.year][r-1]):
        raise Exception("Rounds being incorrectly reviewed")
        
      for g in range(len(self.rounds[r].games)):
        if self.rounds[r].games[g] == years_and_correct_games_for_rounds[self.year][r-1][g]: #correct guess
          points += 1*scoring_scale[r-1]
          if not self.rounds[r].round_name in correct_guesses.keys(): 
            correct_guesses[self.rounds[r].round_name] = 1
          else:
            correct_guesses[self.rounds[r].round_name] = correct_guesses[self.rounds[r].round_name] + 1
            
    print("BRACKET SCORE:")
    print("scale: " + str(scoring_scale))
    print("Total Score: " + str(points))
    total_correct = 0
    for r in range(len(correct_guesses)):
      total_correct += correct_guesses[list(correct_guesses.keys())[r]]
      print(str(correct_guesses[list(correct_guesses.keys())[r]]) + " out of " + list(correct_guesses.keys())[r])
    print("Total Correct: " + str(total_correct))
    
class Round():
  
  def __init__(self, round_name, games):
    self.round_name = round_name
    self.games = games
    
  #simulate_round():
    #for each game
      #simulate game and add to new array
  #return next round
    

