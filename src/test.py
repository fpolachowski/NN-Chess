import pandas as pd
from ast import literal_eval

annotation_file = f"data/train.csv"
with open(annotation_file) as f:
    df = pd.read_csv(f, delimiter=',', encoding='utf-8', header=0)
    
    df["Moves"] = df["Moves"].apply(literal_eval)
    
    _set = set([move for game in df["Moves"] for move in game])
    print(_set)
    token_dict = {move : i for i, move in enumerate(_set)}
    print(token_dict)
