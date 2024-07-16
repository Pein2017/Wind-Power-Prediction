import torch
from torch.utils.data import Dataset


class WindPowerDataset(Dataset):
    def __init__(self, X, y, X_mark, seq_len, pred_len=1):
        self.X = X
        self.y = y
        self.X_mark = X_mark
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.X) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = torch.tensor(self.X[s_begin:s_end], dtype=torch.float32)
        seq_y = torch.tensor(self.y[r_begin:r_end], dtype=torch.float32)
        seq_x_mark = torch.tensor(self.X_mark[s_begin:s_end], dtype=torch.float32)
        seq_y_mark = torch.tensor(self.X_mark[r_begin:r_end], dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark
