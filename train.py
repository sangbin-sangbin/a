import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import torch.nn.functional as F
import random


class Model(nn.Module):
    def __init__(self, input_size, hidden_dim, tagset_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, tagset_size)

    def forward(self, landmarks):
        x = self.fc1(landmarks)
        x = self.fc2(x)
        res = F.log_softmax(x, dim=0)
        return res


INPUT_SIZE = 42
HIDDEN_DIM = 32
TARGET_SIZE = 3

model = Model(INPUT_SIZE, HIDDEN_DIM, TARGET_SIZE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

dataset = json.load(open('dataset.json'))
random.shuffle(dataset)
train_dataset = dataset[:int(len(dataset)*0.7)]
eval_dataset = dataset[int(len(dataset)*0.7):]

print(len(dataset))

for epoch in range(1):
    for data in train_dataset:
        model.zero_grad()

        landmarks = torch.tensor([element for row in data['landmarks'] for element in row], dtype=torch.float)

        ans = [0 for _ in range(TARGET_SIZE)]
        ans[data['gesture']] = 1
        ans = torch.tensor(ans, dtype=torch.float)

        res = model(landmarks)

        loss = loss_function(res, ans)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

good = 0
bad = 0
for data in eval_dataset:
    res = list(model(torch.tensor([element for row in data['landmarks'] for element in row], dtype=torch.float)))
    if data['gesture'] == res.index(max(res)):
        good += 1
    else:
        bad += 1
print(good/(good+bad)*100)


torch.save(model.state_dict(), './model.pt')


