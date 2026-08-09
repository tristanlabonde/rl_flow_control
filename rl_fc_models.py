import torch.nn as nn
import torch


nb_snapshots = 20000
nb_components = 3 # velocity u, velocity v, velocity w
nb_channels = nb_components # channels : velocity u, velocity v, velocity w
ng = [512, 128, 160] # dimensions of the velocity field
starting_x = 100
cutting_rate = 1
control_width = 64//cutting_rate
control_length = ng[1]//cutting_rate
nb_actions = control_width * control_length

class FlowControlGrids(nn.Module): # Predict 1 per time step
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
        
        return self.final(x).reshape(-1, control_width, control_length)


class FlowControlSingleGrid(nn.Module): # Predict only 1 grid to apply at each time step, continuous in time
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

        self.final = nn.Sequential(
            nn.Linear(ng[0]*ng[1]*ng[2]//(1024*cutting_rate), nb_actions),
            nn.Tanh()
        )
        
    def forward(self, velocity_profile, time_vector):

        v = self.conv(velocity_profile)
        v = self.reduce(v)
        
        return self.final(v).reshape(-1, control_width, control_length)

nb_coeffs = 1 # number of Fourier series coefficients to predict

class FlowControlCoeffGrids(nn.Module): # Fourier series coefficients, 1 grid per time step
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
            nn.Linear(ng[0]*ng[1]*ng[2]//(1024*cutting_rate) + 16, nb_coeffs),
            nn.Tanh()
        )

        x_grid = starting_x + torch.linspace(0.0, control_width, control_width)*cutting_rate
        y_grid = torch.linspace(0.0, control_length, control_length)*cutting_rate
        k_modes = torch.arange(1, nb_coeffs + 1, dtype=torch.float32)

        sin_x = torch.sin(2*torch.pi/(control_width * cutting_rate) * x_grid.unsqueeze(0) * k_modes.unsqueeze(1))
        sin_y = torch.sin(2*torch.pi/(control_length * cutting_rate) * y_grid.unsqueeze(0) * k_modes.unsqueeze(1))

        self.register_buffer('sin_x', sin_x)
        self.register_buffer('sin_y', sin_y)
        
    def forward(self, velocity_profile, time_vector):

        v = self.conv(velocity_profile)
        v = self.reduce(v)
        
        num_snapshots = time_vector.size(0)
        v = v.expand(num_snapshots, -1)
        
        t = self.time(time_vector)
        
        x = torch.cat((v, t), dim=1)

        a = self.final(x)
        for c in a:
            w = torch.zeros((control_width, control_length), device=c.device)
            for i in range(nb_coeffs):
                w += c * self.sin_x[i].unsqueeze(1) * self.sin_y[i]
        
        return w

class FlowControlCoeffSingleGrid(nn.Module): # Fourier series coefficients, 1 grid for all time steps, continuous in time
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

        self.final = nn.Sequential(
            nn.Linear(ng[0]*ng[1]*ng[2]//(1024*cutting_rate), nb_coeffs),
            nn.Tanh()
        )

        x_grid = starting_x + torch.linspace(0.0, control_width, control_width)*cutting_rate
        y_grid = torch.linspace(0.0, control_length, control_length)*cutting_rate
        k_modes = torch.arange(1, nb_coeffs + 1, dtype=torch.float32)

        sin_x = torch.sin(2*torch.pi/(control_width * cutting_rate) * x_grid.unsqueeze(0) * k_modes.unsqueeze(1))
        sin_y = torch.sin(2*torch.pi/(control_length * cutting_rate) * y_grid.unsqueeze(0) * k_modes.unsqueeze(1))

        self.register_buffer('sin_x', sin_x)
        self.register_buffer('sin_y', sin_y)
        
    def forward(self, velocity_profile, time_vector):

        v = self.conv(velocity_profile)
        v = self.reduce(v)

        a = self.final(v)
        for c in a:
            w = torch.zeros((control_width, control_length), device=c.device)
            for i in range(nb_coeffs):
                w += c * self.sin_x[i].unsqueeze(1) * self.sin_y[i]
        
        return w
    
class FlowControlPINN(nn.Module): # PINN, doesn't compile
    def __init__(self):
        super().__init__()
        self.branch_net = nn.Sequential(
            nn.Conv3d(nb_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=4),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(ng[0]*ng[1]*ng[2]//64, ng[0]*ng[1]*ng[2]//(1024*cutting_rate)),
            nn.Tanh()
        )

        self.trunk_net = nn.Sequential(
            nn.Linear(ng[0]*ng[1]*ng[2]//(1024*cutting_rate) + 3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        x_grid = torch.linspace(0.0, 1.0, control_width)
        y_grid = torch.linspace(0.0, 1.0, control_length)
        grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        
        self.register_buffer('grid_x', grid_x.reshape(-1, 1))
        self.register_buffer('grid_y', grid_y.reshape(-1, 1))
        
    def forward(self, velocity_profile, time_vector):
        num_times = time_vector.size(0)
        num_wall_points = nb_actions 

        fluid_context = self.branch_net(velocity_profile)
        
        x_expanded = self.grid_x.repeat(num_times, 1)
        y_expanded = self.grid_y.repeat(num_times, 1)
        t_expanded = time_vector.repeat_interleave(num_wall_points, dim=0)
        
        context_expanded = fluid_context.expand(num_times * num_wall_points, -1)
        
        decoder_input = torch.cat([x_expanded, y_expanded, t_expanded, context_expanded], dim=1)
        w_raw = self.trunk_net(decoder_input)
        
        # smooth_mask = torch.sin(torch.pi * x_expanded) * torch.sin(torch.pi * y_expanded)
        # w_continuous = w_raw * smooth_mask
        
        action_grid = w_raw.reshape(num_times, self.Nx_wall, self.Ny_wall)
        
        return action_grid