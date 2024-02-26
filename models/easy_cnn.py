import torch
import torch.nn as nn

class ConvEncoderWithGN(nn.Module):
    def __init__(self):
        super(ConvEncoderWithGN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.relu = nn.ELU()
        # Adjusted stride for conv2 to 43
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=43, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=128)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return x

# Create the model
model = ConvEncoderWithGN()

# Dummy input of shape (batch_size, channels, length)
# Assuming batch size of 1 for illustration
input_data = torch.randn(1, 5, 2048)

# Forward pass
output = model(input_data)

# Output shape should now be torch.Size([1, 128, 48])
print(output.shape)



import torch
import torch.nn as nn

class ConvEncoderWithGN(nn.Module):
    def __init__(self, in_channels, in_len, out_channels, out_len):
        super(ConvEncoderWithGN, self).__init__()
        # Initial convolution to reduce dimensionality
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels // 2, 
                               kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels // 2)
        self.relu = nn.ReLU()
        
        # Second convolution for further processing
        self.conv2 = nn.Conv1d(in_channels=out_channels // 2, out_channels=out_channels, 
                               kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        
        # Adaptive pooling to resize to the desired output length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(out_len)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.adaptive_pool(x)  # Resize feature maps to the desired output length
        return x

# Test the model with specified parameters
in_channels = 5
in_len = 2048
out_channels = 128
out_len = 48

model = ConvEncoderWithGN(in_channels, in_len, out_channels, out_len)
input_data = torch.randn(1, in_channels, in_len)
output = model(input_data)

print(output.shape)




class ConvEncoderWithGN(nn.Module):
    def __init__(self, in_channels, out_channels, out_len):
        super(ConvEncoderWithGN, self).__init__()
        # Initial convolution to reduce dimensionality
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels // 2, 
                               kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels // 2)
        self.elu = nn.ELU()
        
        # Second convolution for further processing
        self.conv2 = nn.Conv1d(in_channels=out_channels // 2, out_channels=out_channels, 
                               kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        
        # Adaptive pooling to resize to the desired output length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(out_len)

    def forward(self, x):
        x = self.elu(self.gn1(self.conv1(x)))
        x = self.elu(self.gn2(self.conv2(x)))
        x = self.adaptive_pool(x)  # Resize feature maps to the desired output length
        return x


# Test the model with specified parameters
in_channels = 5
in_len = 2048
out_channels = 128
out_len = 48

model = ConvEncoderWithGN(in_channels, out_channels, out_len)
input_data = torch.randn(1, in_channels, in_len)
output = model(input_data)

print(output.shape)