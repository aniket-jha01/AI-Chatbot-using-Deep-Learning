import torch
import torch.nn as nn

#THIS IS FOR CREATING A PYTORCH MODEL

#Our feed forward neural net gets the bag of words as the input , then we have one layer fully connected 
#which has the number of patterns as the input size, then the hidden layer, then another hidden layer
#and the output size is equal to the number of classes, then we apply the softmax and get probabilities for each class

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):  
        super(NeuralNet,self).__init__()         #So input size and number of classes is fixed, we can change the hidden size
        self.l1=nn.Linear(input_size,hidden_size) 
        self.l2=nn.Linear(hidden_size, hidden_size)
        self.l3=nn.Linear(hidden_size, num_classes)
        self.relu=nn.ReLU()

    def forward(self,x):          #our forward pass
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        out=self.relu(out)
        out=self.l3(out)
        #no activation after this and it will automatically apply the cross entropy loss
        return out

