from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, output_size=128):
        super(ProjectionHead, self).__init__()

        self.lin1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)

        return out

        
class SimCLR(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimCLR, self).__init__()
        
        # backbone 
        backbone = resnet34(weights=None)
        self.representation = torch.nn.Sequential(*list(backbone.children())[:-1])

        # projection
        self.projection = ProjectionHead(input_size, hidden_size, output_size)

    def forward(self, x):
        out = self.representation(x)
        out = out.reshape(out.shape[0],out.shape[1])
        out = self.projection(out)

        return out

