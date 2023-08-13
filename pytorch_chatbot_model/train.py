import json
from preprocessing import tokenizer, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from chat_model import NeuralNet as network


with open("intent.json", "r") as f:
	raw_data = json.load(f)

all_words = list()
tags = list()
xy = list()

for data in raw_data["intents"]:
	tag = data["intent"]
	tags.append(tag)

	for responce in data["text"]:
		w = tokenizer(responce)
		all_words.extend(w)
		xy.append((w, tag))

ignore_words = ["?", "!", ".", ","]

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


train_data = list()
train_labels = list()


for (pattern_sent, tag) in xy:
	bag = bag_of_words(pattern_sent, all_words)
	train_data.append(bag)

	label = tags.index(tag)
	train_labels.append(label)

train_data = np.array(train_data)
train_labels = np.array(train_labels)

class ChatDataset(Dataset):
	def __init__(self):
		self.n_samples = len(train_data)
		self.train_data = train_data
		self.train_labels = train_labels

	def __getitem__(self, index):
		return self.train_data[index], self.train_labels[index]

	def __len__(self):
		return self.n_samples

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(train_data[0])
learning_rate = 0.001
EPOCHS = 100

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bot = network(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(bot.parameters(), lr = learning_rate)

for epoch in range(EPOCHS):
	for (words, labels) in train_loader:
		words = words.to(device)
		labels = labels.to(device).long()

		outputs = bot(words)
		outputs = outputs
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if (epoch + 1) % 10  == 0:
		print(f"{epoch + 1}/{EPOCHS}, loss = {loss.item():.4f}")

print(f"final loss = {loss.item():.4f}")

print(bot.state_dict)

data = {
	"model_state": bot.state_dict(),
	"input_size": input_size,
	"output_size" : output_size,
	"hidden_size" : hidden_size,
	"all_words": all_words,
	"tags": tags
}

FILE = "data.pth"
##torch.save(data, FILE)

print(f"training_complete, file saved to {FILE}")