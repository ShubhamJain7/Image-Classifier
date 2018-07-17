# Import necessary packages and modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

def get_args():
	# Creating Parser
	parser = argparse.ArgumentParser(description="Train Neural Network")
	# Adding arguments to parser
	parser.add_argument('data_dir',type=str,action='store',help='Path to Image dataset')
	parser.add_argument('--save_dir',type=str,action='store',help='Path to store checkpoint',default='')
	parser.add_argument('--arch',type=str,action='store',help='Choose model from torchvision.models(densenet121 or vgg16)',default='densenet121')
	parser.add_argument('--learn_rate',type=float,action='store',help='Set learn rate for training',default=0.001)
	parser.add_argument('--hidden_units',type=int,action='store',help='Set size of each hidden layer',default=512)
	parser.add_argument('--epochs',type=int,action='store',help='Set number of epochs for training',default=10)	
	parser.add_argument('--gpu',action='store_true',help='Use GPU for training')

	return parser.parse_args()	

# Function to perform validation
def validation(model, testloader, criterion, device):
	test_loss = 0
	accuracy = 0
	for images, labels in testloader:
		images, labels = images.to(device), labels.to(device)

		output = model.forward(images)
		test_loss += criterion(output, labels).item()

		ps = torch.exp(output)
		equality = (labels.data == ps.max(dim=1)[1])
		accuracy += equality.type(torch.FloatTensor).mean()
		
	return test_loss, accuracy

# Function to train the model
def training(model, train_loader, validation_loader, epochs, print_every, criterion, optimizer, device):
	epochs = epochs
	steps = 0
	model = model.to(device)
	print("Model is training....")
	
	for e in range(epochs):
		model.train() #Network in training mode
		running_loss = 0
	
		for i, (images, labels) in enumerate(train_loader):
			steps += 1
			images, labels = images.to(device), labels.to(device)
			
			optimizer.zero_grad()
			outputs = model.forward(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			
			if steps % print_every == 0:
				model.eval() #Network in evaluation mode
				with torch.no_grad():
					validation_loss, accuracy = validation(model, validation_loader, criterion, device)
					
				print("Epoch: {}/{}... ".format(e+1, epochs),
					  "| Training Loss: {:.4f}".format(running_loss / print_every),
					  "| Validation Loss: {:.3f}.. ".format(validation_loss  / len(validation_loader)),
					  "| Validation Accuracy: {:.3f}%".format(accuracy / len(validation_loader) * 100))
				running_loss = 0
				model.train() #Network in training mode
	
	print("Finished Training!")

def main():
	# Get command line arguments
	args = get_args()
	# Extract arguments to variables
	data_dir = args.data_dir
	save_dir = args.save_dir
	arch = args.arch
	learn_rate = args.learn_rate
	hidden_units = args.hidden_units
	epochs = args.epochs
	gpu = args.gpu

	# Setting All parameters
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
	batch_size = 64
	print_every = 20
	
	input_size = 1024 if arch=='densenet121' else 25088
	hidden_size = hidden_units
	output_size = 102

	# Defining transforms for the datasets
	train_transforms = transforms.Compose([transforms.RandomRotation(30),
										   transforms.RandomResizedCrop(224),
										   transforms.RandomHorizontalFlip(),
										   transforms.ToTensor(),
										   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
										 ])
	validation_transforms = transforms.Compose([transforms.Resize(256),
												transforms.CenterCrop(224),
												transforms.ToTensor(),
												transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
											  ])
	test_transforms = transforms.Compose([transforms.Resize(256),
										  transforms.CenterCrop(224),
										  transforms.ToTensor(),
										  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
										])
	data_transforms = {'train':train_transforms , 'validation':validation_transforms , 'test':test_transforms}

	# Loading the datasets
	train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
	validation_data = datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
	test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
	data = {'train':train_data , 'validation':validation_data , 'test':test_data}

	# Defining dataloaders
	train_loader = torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True)
	validation_loader = torch.utils.data.DataLoader(data['validation'], batch_size=batch_size)
	test_loader = torch.utils.data.DataLoader(data['test'], batch_size=batch_size)
	data_loaders = {'train':train_loader , 'validation':validation_loader , 'test':test_loader}

	# Defining model
	model_to_user = getattr(models, arch)
	model = model_to_user(pretrained = True)
	for param in model.parameters():
		param.requires_grad = False


	# Defining custom Classifier
	classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_size)),
											('relu1', nn.ReLU()),
											('dropout1', nn.Dropout(p = 0.5)),
											('fc2', nn.Linear(hidden_size, hidden_size //2)),
											('relu2', nn.ReLU()),
											('dropout2', nn.Dropout(p = 0.2)),
											('fc3', nn.Linear(hidden_size //2, output_size)),
											('output', nn.LogSoftmax(dim = 1))
										  ]))
	model.classifier = classifier

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
	

	# Training the network
	training(model, data_loaders['train'], data_loaders['validation'], epochs, print_every, criterion, optimizer, device)
	

	checkpoint = {"model":model,
				   "state_dict":model.state_dict(),
				   "optimizer_dict":optimizer.state_dict(),
				   "class_to_idx":data['train'].class_to_idx,
				   "classifier":model.classifier,
				   "epochs":epochs,
				   "input_size":input_size,
				   "hidden_size":hidden_size,
				   "output_size":output_size,
	}
	torch.save(checkpoint, './' + save_dir + '/checkpoint.pth')
	print('Checkpoint Saved!')

if __name__ == '__main__':
	main()