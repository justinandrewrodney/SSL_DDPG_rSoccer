import torch
import numpy as np
def hidden_init(layer):
    lim = 1. / np.sqrt(layer.weight.data.size()[0])
    return lim
class ActorMod(torch.nn.Module):
    def __init__(self, state_size, action_size, seed = 0,a_layer1_units=256, a_layer2_units=256):
        super(ActorMod, self).__init__()
        self.a_layer1 = torch.nn.Linear(state_size, a_layer1_units)
        self.a_layer2 = torch.nn.Linear(a_layer1_units, a_layer2_units)
        self.a_layer3 = torch.nn.Linear(a_layer2_units, action_size)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        #self.reset_parameters()
    
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

class CriticMod(torch.nn.Module):
    def __init__(self, state_size, action_size, seed = 0, state_c_layer1_units=16, state_c_layer2_units=32,\
                action_c_layer1_units=32, c_layer1_units=256, c_layer2_units= 256):
        super(CriticMod, self).__init__()
        # self.state_c_layer1 = torch.nn.Linear(state_size, state_c_layer1_units);
              
        self.state_c_layer1 = torch.nn.Linear(state_size , state_c_layer1_units);
        #self.state_c_layer2 = torch.nn.Linear(state_c_layer1_units, state_c_layer2_units);
        
        #self.action_c_layer1 = torch.nn.Linear(action_size, action_c_layer1_units) 
        
        #self.c_layer1 = torch.nn.Linear(state_c_layer2_units +action_c_layer1_units, c_layer1_units);

        self.c_layer1 = torch.nn.Linear(state_c_layer1_units +action_size, c_layer1_units);
        self.c_layer2 = torch.nn.Linear(c_layer1_units, c_layer2_units);
        self.c_layer3 = torch.nn.Linear(c_layer2_units, 1);

        self.relu = torch.nn.ReLU()

    
    # def reset_parameters(self):
    #     state_c_layer1_init = hidden_init(self.state_c_layer1)
    #     torch.nn.init.uniform_(self.state_c_layer1.weight, -state_c_layer1_init, state_c_layer1_init)
        
    #     state_c_layer2_init = hidden_init(self.state_c_layer2)
    #     torch.nn.init.uniform_(self.state_c_layer2.weight, -state_c_layer2_init, state_c_layer2_init)
        
    #     action_c_layer1_init = hidden_init(self.action_c_layer1)
    #     torch.nn.init.uniform_(self.action_c_layer1.weight, -action_c_layer1_init, action_c_layer1_init)
        
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
 
        x_s = self.relu(self.state_c_layer1(x_s));

        x_s = torch.cat((x_s,x_a), 1);

        x_s = self.relu(self.c_layer1(x_s));
        x_s = self.relu(self.c_layer2(x_s));  
        x_s = self.c_layer3(x_s);

        return x_s;


#my_module = MyModule(10,20)
#actor_script = torch.jit.script(Actor(6,3))
#critic_script = torch.jit.script(Critic(6,3))

#actor_script.save("actor_script_model.pt")
#critic_script.save("critic_script_model.pt")



