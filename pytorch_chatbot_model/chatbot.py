import random
import json
import torch
from chat_model import NeuralNet as network
from preprocessing import tokenizer, bag_of_words


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intent.json", "r") as f:
	intents = json.load(f)

FILE = "data.pth"

data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
bot_state = data["model_state"]

bot = network(input_size, hidden_size, output_size)
bot.load_state_dict(bot_state)

bot.eval()

bot_name = "E.L.S.A."
print("Let's chat! type quit to exit")

while True:
	sentence = input("You: ")
	if sentence == "quit":
		break

	sentence = tokenizer(sentence)
	data = bag_of_words(sentence, all_words)
	data = data.reshape(1, data.shape[0])
	data = torch.from_numpy(data)

	output = bot(data)
	_, predicted = torch.max(output, dim = 1)
	tag = tags[predicted.item()]

	probs = torch.softmax(output, dim = 1)
	prob = probs[0][predicted.item()]

	if prob.item() > 0.75:
		for intent in intents["intents"]:
			if tag == intent["intent"]:
				print(f"{bot_name}: {random.choice(intent['responses'])}")


	else:
		print(f"{bot_name}: I do not undestand")


