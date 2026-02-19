from torch import nn
import torch.nn.functional as F

class CNN1dBaseline(nn.Module):

    def __init__(
            self, 
            in_chan: int = 3, #n. of input channels
            hidden: int = 64, #n. of hidden channels
            kernel: int = 5, #kernel size
            dropout: float = 0.1 #dropout probability hyp.
    ):
        """1D-CNN that takes as input 1D-temporal sequence (trajectory positions) [N, 3, T] and returns rho value prediction [N,].\n
           Architecture:
           - Conv1d + BatchNorm + ReLU (x3 blocks)        
        """
        super().__init__()
        
        #convolutional layers
        self.conv1 = nn.Conv1d(in_chan, hidden, kernel, padding = kernel // 2) #padding used to keep channel dimensions equal and to avoid loss of information about borders
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding = kernel // 2)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel, padding = kernel // 2)
        self.relu = nn.ReLU()

        #batch normalization layers 
        self.norm1 = nn.BatchNorm1d(hidden)
        self.norm2 = nn.BatchNorm1d(hidden)
        self.norm3 = nn.BatchNorm1d(hidden)
        
        self.dropout = nn.Dropout(dropout)

        #final regression head
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.regr_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        #reshaping x to the desidered shape: [N, C, T]
        B, T, C = x.shape 
        x = x.view(B, C, T)

        #conv. layers 
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))

        #final regr. head
        x = self.avg_pool(x).squeeze(-1) #avg. pool returns shape [N, C, 1] -> desidered: [N, C] 
        rho = self.regr_head(x).squeeze(-1) #returns shape [N, 1] -> desidered: (N,)

        return rho

