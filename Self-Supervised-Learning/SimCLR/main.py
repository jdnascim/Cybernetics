from utils import dataloader
from simclr import SimCLR
import torch
import torch.nn as nn
import yaml
from yaml.loader import SafeLoader

class NTXent(nn.Module):
    def __init__(self):
        super(NTXent, self).__init__()

    def __l(self, i, j, simmt, temperature=0.5):
        items = simmt.shape[0]
        l1 = torch.exp(simmt[i][j] / temperature) 
        l2 = 0
        for k in range(items):
            if k != i:
                l2 += torch.exp(simmt[i][k] / temperature)
        l2 = -1 * torch.log(l2)

        return l1 / l2
        
 
    def forward(self, feature_mt, temperature=0.5):

        items = feature_mt.shape[0]
        simmt = torch.zeros([items])
        for i in range(items):
            for j in range(items):
                simmt = nn.cosine_similarity(feature_mt[i],
                                             feature_mt[j])

        loss = 0
        for c in range(0, items, 2):
            i = c
            j = c + 1

            loss += self.__l(i, j, simmt, temperature)
            loss += self.__l(j, i, simmt, temperature)

        loss /= items

    
def train(model, epochs, batch_size, output_size, lr):
    model.train()
    model.zero_grad()

    optimizer  = torch.optim.SGD(model.parameters(), lr=lr)
    
    dt1, dt2 = dataloader(batch_size, device)

    for e in range(epochs):

        for aug1, aug2 in zip(dt1, dt2):
            if len(aug1[0]) != len(aug2[0]):
                raise Exception("Error - Batchs differ in size")
            else:
                batch_size = len(aug1[0])
                feature_mt = torch.zeros([batch_size * 2, output_size])

                feature_mt[0:batch_size*2:2] = model(aug1[0].to(device))
                feature_mt[1:batch_size*2:2] = model(aug2[0].to(device))

        loss = NTXent()(feature_mt)

        print("EPOCH: {} LOSS: {}".format(e, loss))

        loss.backward()
        optimizer.step()
        

if __name__ == "__main__":
    # Open the file and load the file
    with open('config.yml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    input_size = config['input_size']
    hidden_size = config['hidden_size']
    output_size = config['output_size']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    device = config['device']

    model = SimCLR(input_size, hidden_size, output_size)

    model.to(device)

    train(model, epochs, batch_size, output_size, lr=learning_rate)