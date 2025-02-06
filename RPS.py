# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random
pattern_counts = {}
game_round = 0
win_rate = 0
wins = 0
decay_factor = 0.85
def player(prev_play, opponent_history=[]):
    global game_round, wins, win_rate
    game_round += 1
    if prev_play:
        opponent_history.append(prev_play)
    else:
        return random.choice(["R","P","S"])    
    
    
   
    if len(opponent_history) < 5:
        return random.choice(["R","P","S"])
    
    
    if game_round % 50 < 25:
        last_moves = tuple(opponent_history[-4:])
    else:
        last_moves = tuple(opponent_history[-3:])
      
    
    if last_moves not in pattern_counts: 
        pattern_counts[last_moves] = {"R":0, "P":0, "S":0}
        

    
    
    for key in pattern_counts[last_moves]:
        pattern_counts[last_moves][key] *= decay_factor
    pattern_counts[last_moves][prev_play] +=1
    

        
    vorhersage = max(pattern_counts[last_moves], key=pattern_counts[last_moves].get)
      
        
    if game_round % 100 == 0:
        if win_rate < 50:
            pattern_length = random.choice([2,3,4,5])    
            last_moves = tuple(opponent_history[-pattern_length:])   
        
    
    move = {"R":"P","P":"S","S":"R"}
    guess = move[vorhersage]
    
    
    if prev_play  == guess:
        wins += 1
    win_rate = (wins / game_round) * 100
    
    return guess
