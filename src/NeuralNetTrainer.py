import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)

        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
        x = torch.sigmoid(self.out(x))
        return x

def train():
    students = pd.read_csv('C:/Users/dimas/CsvGenerator/data/students.csv')
    courses = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')
    data = pd.read_csv('C:/Users/dimas/CsvGenerator/data/student_course_success.csv')

    df = pd.merge(data, students, on='student_id')
    df = pd.merge(df, courses, on='course_id')

    df = pd.get_dummies(df, columns=['major', 'category', 'difficulty_level'])

    feature_names = [col for col in df.columns if col not in ['student_id', 'course_id', 'success']]
    X = df[feature_names]
    y = df['success']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    model = Net(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    print("\n✅ Нейросеть обучена и сохранена!")

    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_names': feature_names
    }, 'C:/Users/dimas/CsvGenerator/src/course_recommender_nn.pt')

if __name__ == '__main__':
    train()
