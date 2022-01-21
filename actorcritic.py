import torch
import numpy as np
def hidden_init(layer):
    lim = 1. / np.sqrt(layer.weight.data.size()[0])
    return lim
class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size,seed=0,a_layer1_units=256,a_layer2_units=256):
        super(Actor, self).__init__()
        self.a_layer1 = torch.nn.Linear(state_size, a_layer1_units)
        self.a_layer2 = torch.nn.Linear(a_layer1_units, a_layer2_units)
        self.a_layer3 = torch.nn.Linear(a_layer2_units, action_size)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
    
    # def reset_parameters(self):
    #     l_1_init = hidden_init(self.a_layer1)
    #     torch.nn.init.uniform_(self.a_layer1.weight, -l_1_init, l_1_init)
        
    #     l_2_init = hidden_init(self.a_layer2)
    #     torch.nn.init.uniform_(self.a_layer2.weight, -l_2_init, l_2_init)

    #     torch.nn.init.uniform_(self.a_layer3.weight, -3e-3, 3e-3)
    
    def forward(self, x_s):
        x_s = self.relu(self.a_layer1(x_s));
        x_s = self.relu(self.a_layer2(x_s));
        x_s = self.a_layer3(x_s);

        x_s = self.tanh(x_s);
        return x_s;

class Critic(torch.nn.Module):
    def __init__(self,state_size,action_size,seed=0,c_layer1_units=256,c_layer2_units=256):
        super(Critic, self).__init__()
        self.c_layer1 = torch.nn.Linear(state_size +action_size, c_layer1_units);
        self.c_layer2 = torch.nn.Linear(c_layer1_units, c_layer2_units);

        self.c_layer3 = torch.nn.Linear(c_layer2_units, 1);

        self.relu = torch.nn.ReLU()

    
    # def reset_parameters(self):
    #     l_1_init = hidden_init(self.c_layer1)
    #     torch.nn.init.uniform_(self.c_layer1.weight, -l_1_init, l_1_init)
        
    #     l_2_init = hidden_init(self.c_layer2)
    #     torch.nn.init.uniform_(self.c_layer2.weight, -l_2_init, l_2_init)

    #     torch.nn.init.uniform_(self.c_layer3.weight, -3e-3, 3e-3)

    def forward(self, x_s, x_a):
        if (x_s.dim() == 1):
            x_s = torch.unsqueeze(x_s, 0);

        if (x_a.dim() == 1):
            x_a = torch.unsqueeze(x_a,0);

        x_s = torch.cat((x_s,x_a), 1);
 
        x_s = self.relu(self.c_layer1(x_s));
        x_s = self.relu(self.c_layer2(x_s));  
        x_s = self.c_layer3(x_s);

        return x_s;




