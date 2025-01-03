import chess.pgn
import pandas as pd
from tqdm import tqdm

NUM_GAMES = 15000

pgn = open("data/lichess_db_standard_rated_2013-01.pgn")

move_dict = set()
games = []

for i in tqdm(range(NUM_GAMES), total=NUM_GAMES):
    game = chess.pgn.read_game(pgn)
    if game is None:
        break
    
    _moves = game.mainline_moves()
    board = game.board()
    
    game_moves = []
    # legal_moves = []
    for move in _moves:
        # legal_moves.append([str(move) for move in board.generate_legal_moves()])
        game_moves.append(str(move))
        move_dict.add(str(move))
    
    
    game_data = [
        game.headers.get("Result"),
        game_moves,
        # legal_moves
    ]
    games.append(game_data)

# I want to separate the moves into train and test sets and save them as test.csv and train.csv
# I will also save the moves and the winner as a csv file
with open("data/test.csv", "w") as f:
    pd.DataFrame(games[:NUM_GAMES//10]).to_csv(f, index=False, header=["Winner", "Moves"])
    
with open("datatrain.csv", "w") as f:
    pd.DataFrame(games[NUM_GAMES//10:]).to_csv(f, index=False, header=["Winner", "Moves"])
    
# print(move_dict)
print(len(move_dict))