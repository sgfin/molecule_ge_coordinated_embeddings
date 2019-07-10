import torch
import torch.nn as nn

class VanillaSiameseNetL1000(nn.Module):
    def __init__(self):
        super(VanillaSiameseNetL1000, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(978, 400),
                                 nn.SELU(),
                                 nn.Linear(400, 100),
                                 nn.SELU(),
                                 nn.Linear(100, 2)
        )

    def forward(self, input):
        input1, input2 = input
        embed1 = self.mlp(input1)
        embed2 = self.mlp(input2)
        return (embed1, embed2)


