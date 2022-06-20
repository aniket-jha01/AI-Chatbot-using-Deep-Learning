# import parallelTestModule

# if __name__ == '__main__':    
#     extractor = parallelTestModule.ParallelExtractor()
#     extractor.runInParallel(numProcesses=2, numThreads=4)

import json 
import numpy as np
from nltk_utilis import tokenize,stemm,bag_of_words
with open('intents.json','r') as f:
    intents=json.load(f)

#THIS IS FOR CREATING THE DATASET FOR TRAINING

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

#import our pytorch model
from model import NeuralNet

all_the_words=[]

tags=[] #for all different patterns and their texts
xy=[]  #will hold our patterns and their texts

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_the_words.extend(w)   #We use extend and not append because we dont want an array of arrays ,we just want the words
        xy.append((w,tag))

ignored_stuff=['?','!',',','.']
#print(all_the_words)

#Let's apply stemming
all_the_words=[stemm(w) for w in all_the_words if w not in ignored_stuff]
#print(all_the_words)
all_the_words=sorted(set(all_the_words)) #Unique and sorted
tags=sorted(set(tags))

#print(tags)
#Let's create the training data or bag of words

x_train=[]
y_train=[]

for (pattern_sentence,tag) in xy:
     bag=bag_of_words(pattern_sentence,all_the_words)
     x_train.append(bag)

     label=tags.index(tag)
     y_train.append(label)

x_train=np.array(x_train)
y_train=np.array(y_train)


#pytorch code
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train
    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples

#Hyperparameters
batch_size = 8
hidden_size=8
output_size=len(tags)
input_size=len(all_the_words)  #or len(x_train[0])  
learning_rate=0.001  
number_of_epochs=1000

dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

#check the sizes
# print(input_size,len(all_the_words))
# print(output_size,len(tags))


#We created this class so we can automatically iterate over this and get better training           

#We imported our model, now we can initilize our model here


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #checking if i have gpu support otherwise use the gpu
model=NeuralNet(input_size, hidden_size, output_size).to(device)


#loss and optimiser
criterion=nn.CrossEntropyLoss()
optimiser=torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(number_of_epochs):
    for (words,labels) in train_loader:
        words=words.to(device)
        labels = labels.type(torch.LongTensor)
        labels=labels.to(device)

        #forward pass
        outputs=model(words)
        loss=criterion(outputs,labels.long())

        #backward pass and optimizer use step
        #we have to empty the gradients first
        optimiser.zero_grad()
        loss.backward()  #to calculate the back proagation
        optimiser.step()

        #this was our training loop
        
    if(epoch+1)%100==0:
        print(f'epoch{epoch+1}/{number_of_epochs},loss={loss.item():.4f}')

print(f'Loss final, loss={loss.item():.4f} ')

data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_the_words":all_the_words,
    "tags":tags

}

FILE="data.pth"
torch.save(data,FILE)  #will serialise and save it to a file

print(f'Training Complete. File saved to {FILE}')