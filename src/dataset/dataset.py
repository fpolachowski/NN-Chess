import torch
import random
import pandas as pd
from ast import literal_eval
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from multiprocessing import Pool

class ChessGame():
    def __init__(self, data):
        self._data = data

    @property
    def moves(self):
        return self._data["Moves"]

    @property
    def legal_moves(self):
        return self._data["Legal Moves"]
    
    @property
    def winner(self):
        if self._data["Winner"] == "1-0":
            return 0
        elif self._data["Winner"] == "0-1":
            return 1
        return -1 # draws


class ChessDataset(Dataset):
    def __init__(self, split, token_dict=None):
        self.split = split
        self.token_dict = token_dict
        self.dataset_name = f"Chess {self.split} Dataset"
        annotation_file = f"data/{self.split}.csv"
        self.df = loadDatasetDescription(annotation_file)
        self.updateDF()
        
        with Pool(processes=8) as pool:
            self.game_list = pool.starmap(createChessGame, [(row,) for _, row in self.df.iterrows()])

    def __len__(self):
        return len(self.df)
    
    def updateDF(self):
        self.df["Moves"] = self.df["Moves"].apply(lambda x: literal_eval(x))
        # self.df["Legal Moves"] = self.df["Legal Moves"].apply(lambda x: literal_eval(x))
        if self.token_dict is None:
            move_set = set([move for game in self.df["Moves"] for move in game])
            self.token_dict = {move : i for i, move in enumerate(move_set)}

    def __getitem__(self, idx):
        
        it = self.game_list[idx]
        src = [self.token_dict[move] for move in it.moves] # Source sequence

        tgt_input = src[:-1]  # Remove the last token for input
        tgt_output = src[1:]  # Shifted target sequence for output
        
        return {
            "src_input": torch.tensor(src),
            "tgt_input": torch.tensor(tgt_input),
            "tgt_output": torch.tensor(tgt_output)
        }
        
def createChessGame(row):
    return ChessGame(row)
    
def loadDatasetDescription(annotation_file):
    with open(annotation_file) as f:
        df = pd.read_csv(f, delimiter=',', encoding='utf-8', header=0)
        return df
    
def generate_batch(data_batch):
    src_input = pad_sequence([element["src_input"] for element in data_batch], batch_first=True, padding_value=0)
    tgt_input = pad_sequence([element["tgt_input"] for element in data_batch], batch_first=True, padding_value=0)
    tgt_output = pad_sequence([element["tgt_output"] for element in data_batch], batch_first=True, padding_value=0)

    return {
            "src_input": src_input,
            "tgt_input": tgt_input,
            "tgt_output": tgt_output
        }

def get_dataloader(dataset, batch_size, dev):
    """ Get a dataloader from a dataset.
    """
    shuffle = True if dataset.split == "train" else False
    drop_last = True if dataset.split == "train" else False
    dataloader = DataLoader(
        dataset if not dev else Subset(dataset, random.sample(range(len(dataset)), batch_size * 50)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=True,  # set True when loading data on CPU and training on GPU
        collate_fn=generate_batch
    )
    return dataloader
    
def prepare_data(batch_size=32, debug=False):
    """ Prepare data.
    """
    train_dataset = ChessDataset("train")
    test_dataset = ChessDataset("test", token_dict=train_dataset.token_dict)
    train_dl = get_dataloader(train_dataset, batch_size, debug)
    test_dl = get_dataloader(train_dataset, batch_size, True)
    eval_dl = get_dataloader(test_dataset, batch_size, debug)

    return {
        "train_dl": train_dl,
        "test_dl": test_dl,
        "eval_dl": eval_dl,
        "token_dict": train_dataset.token_dict
    }
    
if __name__ == "__main__":
    ds = ChessDataset("train")
    it = ds[5]
    
    print(it)