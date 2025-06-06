import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
torch.serialization.add_safe_globals([StandardScaler])

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
        return torch.sigmoid(self.out(x))

def train():
    # ВНИМАНИЕ: gpa и grade теперь в 100-балльной системе (gpa: 40-100, grade: 0-100)
    # Все пороги успеха и генерация данных должны соответствовать этой шкале
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix

    # Загрузка обучающего датасета
    df = pd.read_csv('C:/Users/dimas/CsvGenerator/data/training_data.csv')

    # One-hot для категориальных
    df = pd.get_dummies(df, columns=['major', 'course_category', 'course_difficulty'])

    # Балансировка классов
    class_0 = df[df['success'] == 0]
    class_1 = df[df['success'] == 1]
    min_len = min(len(class_0), len(class_1))
    df = pd.concat([class_0.sample(min_len, random_state=42),
                    class_1.sample(min_len, random_state=42)])

    # Отделяем признаки и целевую переменную
    feature_names = [col for col in df.columns if col not in ['student_id', 'course_id', 'success']]
    X = df[feature_names]
    y = df['success']

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение на тренировочную и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Перевод в тензоры
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    # Модель и обучение
    model = Net(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    print("\n✅ Нейросеть обучена!")

    # Валидация
    model.eval()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_val_tensor).numpy()
        predicted_labels = (predictions > 0.5).astype(int)

    print("\n📊 Classification Report:")
    print(confusion_matrix(y_val, predicted_labels))
    print(classification_report(y_val, predicted_labels, digits=3))

    # Сохранение модели и скейлера
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_names': feature_names
    }, 'C:/Users/dimas/CsvGenerator/src/course_recommender_nn.pt')

    print("✅ Модель сохранена!")



if __name__ == '__main__':
    train()
