import random 
import json
import torch
from model import NeuralNet
from nltk_utilis import bag_of_words,tokenize

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #checking if i have gpu support otherwise use the gpu

with open('intents.json','r') as f:
    intents=json.load(f)


FILE="data.pth"
data=torch.load(FILE)



input_size=data["input_size"]    #calling values
hidden_size=data["hidden_size"]
output_size=data["output_size"]
tags=data['tags']
all_the_words=data['all_the_words']
model_state=data["model_state"]

model=NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)  #So now it knows our learn parameters

#Moving to evaluation
model.eval()

#Let's implement our actual chat

bot_name="Kittu"

def get_response(message):
    print("Hey ! Let's Chat! type 'exit' to exit ")


    sentence=tokenize(message)
    X=bag_of_words(sentence, all_the_words)
    X=X.reshape(1,X.shape[0])   #our model expects it in this format
    X = torch.from_numpy(X).to(device)    #converting it into a torch tensor

    output=model(X)
    _,predicted=torch.max(output,dim=1)  #We retrieve the predicted info
    
    tag=tags[predicted.item()]  #the class label predicted.item() goes into out file and locates identfies the tag
    #The we want to find the corresponding intent, so we loop over all the intents and check if the tag matches
    
    #A good thing to check is if the probability for this tag is high enough (think )
    
    probabs=torch.softmax(output, dim=1)
    probab=probabs[0][predicted.item()]

    if(probab.item()>0.75):
        for intent in intents["intents"]:
            if tag==intent["tag"]:
                return random.choice(intent['responses'])
            
    
    return "I am sorry , I did not understand... Still learning" 
       
    




    


