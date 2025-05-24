import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib

# 🔧 Гиперпараметры
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# 🧠 Модель
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 🔄 Загрузка и подготовка данных
def load_and_prepare_data():
    students = pd.read_csv('C:/Users/dimas/CsvGenerator/data/students.csv')
    courses = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')
    enrollments = pd.read_csv('C:/Users/dimas/CsvGenerator/data/enrollments.csv')

    df = pd.merge(enrollments, students, on='student_id')
    df = pd.merge(df, courses, on='course_id')

    df['high_grade'] = (df['grade'] >= 4).astype(int)

    df = pd.get_dummies(df, columns=['major', 'category', 'difficulty_level'])

    X = df.drop(columns=['high_grade', 'grade', 'student_id', 'course_id', 'name', 'semester'])
    y = df['high_grade']

    return X, y

# 🧠 Обучение модели
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Net(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Оцениваем на валидации
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss = criterion(val_preds, torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1))
            val_acc = ((val_preds > 0.5).float() == torch.tensor(y_val.values).unsqueeze(1)).float().mean()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.2f}")

    return model, scaler, X.columns.tolist()

# 💾 Сохраняем модель
def save_model(model, scaler, feature_names, filename='course_recommender_nn.pt'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': feature_names,
        'scaler': scaler
    }, filename)
    print(f"✅ Нейросеть обучена и сохранена в '{filename}'")

# 🚀 Точка входа
if __name__ == '__main__':
    print("🔄 Загружаем и подготавливаем данные...")
    X, y = load_and_prepare_data()

    print("🧠 Начинаем обучение нейросети...")
    model, scaler, feature_names = train_model(X, y)

    print("💾 Сохраняем модель...")
    save_model(model, scaler, feature_names)
