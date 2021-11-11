import os
import shutil
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as nnf

import pytorch_cnn_model as pcm 


data_dir = "../../GenX"

model_directory = "Trained_models"
model_name = "model_trial_pokemon_3_2.pth"
model_path = model_directory + "/" + model_name

image_directory = "../../test_images"
image_name = "test6.jpg"
image_path = image_directory + "/" + image_name

if os.path.exists('../../GenX/._.DS_Store'):
	os.remove('../../GenX/._.DS_Store')
if os.path.exists('../../GenX/.DS_Store'):
	os.remove('../../GenX/.DS_Store')
else:
	pass

pokemon = os.listdir(data_dir)
pokemon.sort()


model = torchvision.models.vgg16(pretrained=True)

# for param in model.parameters():
#     param.requires_grad = False

model.features[0] = nn.Conv2d(3,64,kernel_size=(3,3), stride=(1,1), padding=(1,1))
model.classifier[6] = nn.Linear(4096,3)

model.load_state_dict(torch.load(model_path))


IMG_SIZE = 80

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

plt.imshow(X[0].view(IMG_SIZE ,IMG_SIZE ,3))
plt.title(f"It's a {pokemon[predicted_class]} with a confidence of {confidence}%")
plt.show()