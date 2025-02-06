# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random
pattern_counts = {}
game_round = 0
win_rate = 0
wins = 0
decay_factor = 0.81
pattern_length = 4
bot_history = []

def predict_next_moves(oppent_history, steps=2):
    predictions = []
    last_moves = tuple(oppent_history[-pattern_length:])
    
    for _ in range(steps):
        if last_moves in pattern_counts:
            next_move = max(pattern_counts[last_moves], key=pattern_counts[last_moves].get)
        else:
            next_move = random.choice(["R","P","S"])
        predictions.append(next_move)
        last_moves = (*last_moves[1:], next_move)
    return predictions

def player(prev_play, opponent_history=[]):
    global game_round, wins, win_rate,pattern_length
    game_round += 1
    if prev_play:
        opponent_history.append(prev_play)
    else:
        return random.choice(["R","P","S"])    
    if game_round % 300 == 0 and win_rate < 50:
        pattern_length = random.choice([2,3,4,5,6,7])
    last_moves = tuple(opponent_history[-pattern_length:])    
    
    if last_moves not in pattern_counts: 
        pattern_counts[last_moves] = {"R":0, "P":0, "S":0}
   
    for key in pattern_counts[last_moves]:
        pattern_counts[last_moves][key] *= decay_factor
    pattern_counts[last_moves][prev_play] +=1
    
    
        
    vorhersage = max(pattern_counts[last_moves], key=pattern_counts[last_moves].get)
  
    move = {"R":"P","P":"S","S":"R"}
    
    
    strategies = ["pattern_recognition", "random", "mirror"]
    current_strategy = "pattern_recognition"
    if win_rate < 61 and game_round % 250 == 0:
        current_strategy = random.choice(strategies)
    
    if current_strategy == "pattern_recognition":
        guess = move[vorhersage]
    elif current_strategy == "random":
        guess = random.choice(["R","P","S"])
    elif current_strategy == "mirror":
        guess = prev_play if prev_play else random.choice(["R","P","S"])
        

    bot_history.append(guess)
    if len(bot_history) > 1:
        last_bot_play = bot_history[-2]
        winning_moves = {"R": "S", "P": "R", "S": "P"}
        if winning_moves[last_bot_play] == prev_play:
            wins += 1
    win_rate = (wins / game_round) * 100
    
    return guess
