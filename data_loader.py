import numpy as np
import os
from glob import glob
from torch.utils.data import DataLoader


def rand_equation(num_data, save_path):
    for i in range(num_data):
        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)
        c = np.random.randint(1, 10)
        y = a + b*0.1 + c*0.01
        data = np.array([a, b, c, y])
        np.save(os.path.join(save_path, f"{i}.npy"), data)


def make_data(data_path, num_train, num_val, num_test):
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")
    test_path = os.path.join(data_path, "test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    rand_equation(num_train, train_path)
    rand_equation(num_val, val_path)
    rand_equation(num_test, test_path)


class load_dataset():
    def __init__(self, data_path):
        self.data_path = data_path
        self.len_data = len(glob(f"{self.data_path}/*.npy"))
        self.x, self.y = self.load_data(self.data_path)


    def __len__(self):
        return self.len_data


    def load_data(self, path):
        _data = []
        for data in glob(f"{path}/*.npy"):
            data = np.load(data)
            _data.append(data)
        _data = np.concatenate(_data, axis=0).reshape(-1, 4)
        value = _data[:, :3].reshape(-1, 3)
        target = _data[:, -1].reshape(-1, 1)
        return value, target


    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x, y


def get_loader_segment(data_path, mode, batch_size):
    dataset = load_dataset(data_path)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader