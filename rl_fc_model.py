import torch.nn as nn
import torch


nb_snapshots = 20000
nb_components = 3 # velocity u, velocity v, velocity w
nb_channels = nb_components # channels : velocity u, velocity v, velocity w
ng = [512, 128, 160] # dimensions of the velocity field
starting_x = 100
cutting_rate = 4
control_width = 64//cutting_rate
control_length = ng[1]//cutting_rate
nb_actions = control_width * control_length

class RL_FlowControl_Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(nb_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=4),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Flatten()
        )
        
        self.reduce = nn.Sequential(
            nn.Linear(ng[0]*ng[1]*ng[2]//64, ng[0]*ng[1]*ng[2]//(1024*cutting_rate)),
            nn.ReLU()
        )

        self.time = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(ng[0]*ng[1]*ng[2]//(1024*cutting_rate) + 16, nb_actions),
            nn.Tanh()
        )
        
    def forward(self, velocity_profile, time_vector):

        v = self.conv(velocity_profile)
        v = self.reduce(v)
        
        num_snapshots = time_vector.size(0)
        v = v.expand(num_snapshots, -1)
        
        t = self.time(time_vector)
        
        x = torch.cat((v, t), dim=1)
        
        return self.final(x)
