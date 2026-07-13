import torch.nn as nn


nb_snapshots = 3
nb_components = 3 # velocity u, velocity v, velocity w
nb_channels = nb_components # channels : velocity u, velocity v, velocity w
ng = [8, 8, 8] # dimensions of the velocity field
nb_actions = 10

class RL_FlowControl_Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(nb_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(32*ng[0]//4*ng[1]//4*ng[2]//4, ng[0]//4),
            nn.ReLU(),
            nn.Linear(ng[0]//4, nb_actions),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)
