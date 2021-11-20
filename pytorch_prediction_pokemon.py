import os
import shutil
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as nnf
from pathlib import *
#import pytorch_cnn_model as pcm 


def make_list(data_dir, dirs_to_ignore):
	''' 
    '''
	pokemon = []

	for dirs in os.listdir(data_dir):
		if dirs in dirs_to_ignore:
			continue
		else:
			pokemon.append(dirs)
			pokemon.sort()
	return pokemon

def net(model_path):
	''' 
    '''
	model = torchvision.models.vgg16(pretrained=True)

	for param in model.parameters():
		param.requires_grad = False

	model.features[0] = nn.Conv2d(3,64,kernel_size=(3,3), stride=(1,1), padding=(1,1))
	model.classifier[6] = nn.Linear(4096,3)

	model.load_state_dict(torch.load(model_path))

	return model

def image_predictor(net,image_path, IMG_SIZE):
	'''
    '''
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	X = torch.Tensor(img).view(-1,3,IMG_SIZE ,IMG_SIZE )
	X = X/255.0

	with torch.no_grad():
		net_out = model(X.view(-1, 3, IMG_SIZE , IMG_SIZE ))[0]
		predicted_class = torch.argmax(net_out)

		sm = torch.nn.Softmax(dim = -1)
		probabilities = sm(net_out) 

		confidence = torch.round(probabilities[predicted_class]*100)
	return X, predicted_class, confidence

#####################################################################

data_dir = PurePath("../PykaDex_Data/Data/Images/GenX")
model_path = "../PykaDex_Trainer/Trained_models/model_trial_pokemon_3_2.pth"
image_path = "../PykaDex_Data/Data/Test_Images/test.png"
dirs_to_ignore = ['backgrounds','README.md', ".DS_Store", "._.DS_Store"]
IMG_SIZE = 80

pokemon = make_list(data_dir, dirs_to_ignore)
model = net(model_path)
X, predicted_class, confidence = image_predictor(net,image_path, IMG_SIZE)

plt.imshow(X[0].view(IMG_SIZE ,IMG_SIZE ,3))
plt.title(f"It's a {pokemon[predicted_class]} with a confidence of {confidence}%")
plt.show()