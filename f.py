import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Blueprint of the Person
class Person:
    def __init__(self, attractive):
        self.attractive = attractive

# Blueprint of our agent
class SimpleReflexAgent:
    def __init__(self, person):
        self.person = person

    def process_person(self):
        return int(self.person.attractive)

# Neural Network using PyTorch and Keras
class AttractivenessClassifier(nn.Module):
    def __init__(self):
        super(AttractivenessClassifier, self).__init__()
        self.fc = nn.Linear(1, 1)  # 1 input feature, 1 output

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x

# CREATE objects
p1 = Person(attractive=False)
p2 = Person(attractive=True)
p3 = Person(attractive=True)

# CREATE our queue
queue = [p1, p2, p3]

# Neural Network training
# Assuming attractiveness is the only feature
X_train = torch.tensor([[int(p.attractive)] for p in queue], dtype=torch.float32)
y_train = torch.tensor([[1.0] if p.attractive else [0.0] for p in queue], dtype=torch.float32)

model = AttractivenessClassifier()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()


class GUI(tk.Tk):
    def __init__(self, queue):
        super(GUI, self).__init__()

        self.title("Attractiveness Classifier")
        self.geometry("300x150")

        self.queue = queue
        self.agent = SimpleReflexAgent(self.queue[0])

        self.label = tk.Label(self, text="Current Person:")
        self.label.pack(pady=10)

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)

        self.next_button = tk.Button(self, text="Next Person", command=self.next_person)
        self.next_button.pack(pady=10)

    def next_person(self):
        if self.queue:
            current_person = self.queue.pop(0)
            self.agent = SimpleReflexAgent(current_person)
            result = self.agent.process_person()

            
            self.label.config(text=f"Current Person: {current_person.attractive}")
            self.result_label.config(text=f"Allowed: {bool(result)}")

         
            if bool(result):
                messagebox.showinfo("Result", "Person is allowed into the club")
            else:
                messagebox.showinfo("Result", "Person is not allowed into the club")
        else:
            self.label.config(text="No more people in the queue")
            self.result_label.config(text="")

gui = GUI(queue)
gui.mainloop()
