import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from NeuralNetTrainer import Net
import joblib

torch.serialization.add_safe_globals([StandardScaler])

def load_model(filename='C:/Users/dimas/CsvGenerator/src/course_recommender_nn.pt'):
    checkpoint = torch.load(filename, weights_only=False)  # <-- ВАЖНО!
    model = Net(len(checkpoint['feature_names']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    scaler = checkpoint['scaler']
    feature_names = checkpoint['feature_names']
    return model, scaler, feature_names

def recommend_courses_for_student(student_data: dict, all_courses: pd.DataFrame, model, scaler, feature_names: list):
    recommendations = []

    for _, course in all_courses.iterrows():
        # Копируем данные студента
        input_data = student_data.copy()

        # Добавляем признаки курса
        input_data['category_' + course['category']] = 1
        input_data['difficulty_level_' + course['difficulty_level']] = 1
        input_data['credits'] = course['credits']

        # Приводим к правильному порядку признаков
        input_df = pd.DataFrame([input_data], columns=feature_names).fillna(0)

        # Нормализуем с тем же scaler'ом, что использовался при обучении
        input_scaled = scaler.transform(input_df)

        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Предсказание
        with torch.no_grad():
            proba = model(input_tensor).item()

        if proba > 0.5:
            recommendations.append((course['name'], proba))

    # Сортируем по вероятности
    recommendations.sort(key=lambda x: x[1], reverse=True)

    print("\n🤖 Нейросеть рекомендует:")
    for name, prob in recommendations:
        print(f"{name} → Вероятность успеха: {prob:.2f}")

# 🧪 Пример запуска
if __name__ == '__main__':
    # Загружаем модель и scaler
    model, scaler, feature_names = load_model()

    # Пример данных студента (можно взять из students.csv)
    student_example = {
        'gpa': 4.2,
        'year_of_study': 2,
        'major_CS': 1,
        'major_History': 0,
        'major_Math': 0,
        'major_Physics': 0,
        # Остальное будет заполнено позже
    }

    # Загружаем курсы
    courses = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')

    # Получаем рекомендации
    recommend_courses_for_student(student_example, courses, model, scaler, feature_names)
