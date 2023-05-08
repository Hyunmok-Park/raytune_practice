from torch import nn

class Net(nn.Module):
    def __init__(self, l1=64, l2=64):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x