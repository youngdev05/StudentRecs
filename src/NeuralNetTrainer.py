import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from ModelTrainer import load_data, prepare_data

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train():
    students, courses, enrollments = load_data()
    X, y = prepare_data(students, courses, enrollments)

    X = X.astype(np.float32)
    y = y.astype(np.float32).values.reshape(-1, 1)

    model = Net(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X.values)
    y_tensor = torch.tensor(y)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # Сохраняем модель и фичи
    torch.save({'model_state_dict': model.state_dict(), 'feature_names': X.columns.tolist()}, 'course_recommender_nn.pt')
    print("✅ Нейросеть обучена и сохранена!")

if __name__ == "__main__":
    train()
