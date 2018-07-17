# Import necessary packages and modules
import torch
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import json

# Get arguments from command line
def get_args():
	# Creating Parser
	parser = argparse.ArgumentParser(description="Predict species of flower")
	# Adding arguments to parser
	parser.add_argument('path_to_image',type=str,action='store',help='Path to Image of flower to be recognized')
	parser.add_argument('checkpoint',type=str,action='store',help='Path to saved checkpoint',default='')
	parser.add_argument('--top_k',type=int,help='Number of top classes to display',default=5)
	parser.add_argument('--category_names',type=str,help='Path to JSON file containing class labels')
	parser.add_argument('--gpu',action='store_true',help='Use GPU for training')

	return parser.parse_args()

# Loading Checkpoint and rebuilding model
def load_checkpoint(path):
	checkpoint = torch.load(path)
	model = checkpoint['model']
	model.load_state_dict(checkpoint['state_dict'])
	model.class_to_idx = checkpoint['class_to_idx']
	model.classifier = checkpoint['classifier']
	optimizer = checkpoint['optimizer_dict']
	epochs = checkpoint['epochs']
	return model

# Process image before recognizing
def process_image(image):
	im = Image.open(image)
	im = im.resize((256,256))
	im = im.crop((16,16,240,240))
	np_im = np.array(im)
	im_norm = ((np_im / 255) - ([0.485, 0.456, 0.406])) / ([0.229, 0.224, 0.225])
	im_norm = im_norm.transpose((2,0,1))
	return im_norm

# Predict the classes of an image using a trained deep learning model
def predict(image_path, model, device, topk=5):
	image = torch.from_numpy(process_image(image_path))
	image = image.unsqueeze(0).float()
	model, image = model.to(device), image.to(device)
	model.eval() # Setting model to evaluation mode
	model.requires_grad = False
	outputs = torch.exp(model.forward(image)).topk(topk)
	probabilities, classes = outputs[0].data.cpu().numpy()[0], outputs[1].data.cpu().numpy()[0]
	idx_to_class = {key: value for value, key in model.class_to_idx.items()}
	classes = [idx_to_class[classes[i]] for i in range(classes.size)]
	return probabilities, classes

# Display results
def view_classify(im, probs, classes, cat_to_name):
	if cat_to_name is None:
		name_classes = classes
	else:
		with open(cat_to_name, 'r') as f:
			cat_to_name_data = json.load(f)
		name_classes = [cat_to_name_data[i] for i in classes]
	df = pd.DataFrame({
		'classes': pd.Series(data = name_classes),
		'values': pd.Series(data = probs, dtype='float64')
	})
	print(df)

def main():
	# Get command line arguments
	args = get_args()
	# Extract arguments to variables
	path_to_image = args.path_to_image
	checkpoint = args.checkpoint
	top_k = args.top_k
	category_names = args.category_names
	gpu = args.gpu

	# Setting all parameters
	device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
	# Loading model from checkpoint
	model = load_checkpoint(checkpoint)
	# Find top 5 matches and their labels
	probs, classes = predict(path_to_image, model, device, top_k)
	view_classify(path_to_image, probs, classes, category_names)

if __name__ == '__main__':
	main()