import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self, in_channels, in_wh, out_channels_1, out_channels_2, fc_loc_out, kernel_size, stride, padding):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.loc_net = nn.Sequential(
            # First conv. layer.
            nn.Conv2d(in_channels, out_channels_1, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second conv. layer.
            nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        out_size = (((in_wh - kernel_size + 1) // 2 + padding)  - kernel_size + 1) // 2 + padding
        self.fc_loc_in = out_channels_2 * out_size * out_size

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_loc_in, fc_loc_out),
            nn.BatchNorm1d(fc_loc_out),
            nn.ReLU(),
            nn.Linear(fc_loc_out, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x_loc = self.loc_net(x)
        x_loc = x_loc.view(-1, self.fc_loc_in)
        theta = self.fc_loc(x_loc)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

class Net(nn.Module):
    # 2-STN modules.
    # 3-Conv2d layers.
    def __init__(self, in_channels, in_wh, fc1_dim, num_classes, conv_params):
        super(Net, self).__init__()

        self.stn1 = STN(in_channels, in_wh, conv_params['stn_ch1'][0], conv_params['stn_ch1'][1], fc_loc_out = 200, kernel_size = 5, stride = 1, padding = 2)
        self.conv1 = nn.Conv2d(in_channels, conv_params['out_channels'][0], conv_params['kernel_size'][0], conv_params['stride'][0], conv_params['padding'][0])
        self.bn1 = nn.BatchNorm2d(conv_params['out_channels'][0])

        self.conv2 = nn.Conv2d(conv_params['out_channels'][0], conv_params['out_channels'][1], conv_params['kernel_size'][1], conv_params['stride'][1], conv_params['padding'][1])
        self.bn2 = nn.BatchNorm2d(conv_params['out_channels'][1])
        self.stn2 = STN(conv_params['out_channels'][1], 12, conv_params['stn_ch2'][0], conv_params['stn_ch2'][1], fc_loc_out = 150, kernel_size = 5, stride = 1, padding = 2)

        self.conv3 = nn.Conv2d(conv_params['out_channels'][1], conv_params['out_channels'][2], conv_params['kernel_size'][2], conv_params['stride'][2], conv_params['padding'][2])
        self.bn3 = nn.BatchNorm2d(conv_params['out_channels'][2])
        
        self.fc_dim = conv_params['out_channels'][2] * 6 * 6

        self.fc1 = nn.Linear(self.fc_dim, fc1_dim)
        self.bn4 = nn.BatchNorm1d(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, num_classes)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.max_pool2d(F.relu(self.bn1(self.conv1(self.stn1(x)))), 2)
        x = torch.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = torch.max_pool2d(F.relu(self.bn3(self.conv3(self.stn2(x)))), 2)
        x = x.view(-1, self.fc_dim)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(self.drop(x))
        return F.log_softmax(x, dim=1)