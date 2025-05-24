import torch
import pandas as pd
import numpy as np
from NeuralNetTrainer import Net

def load_model(filename='C:/Users/dimas/CsvGenerator/src/course_recommender_nn.pt'):
    checkpoint = torch.load(filename)
    model = Net(len(checkpoint['feature_names']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['feature_names']

def recommend_courses_for_student(student_data, all_courses, model, feature_names):
    recommendations = []

    for _, course in all_courses.iterrows():
        input_data = student_data.copy()
        input_data['category_' + course['category']] = 1
        input_data['difficulty_level_' + course['difficulty_level']] = 1
        input_data['credits'] = course['credits']

        input_df = pd.DataFrame([input_data.reindex(feature_names, fill_value=0)])
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

        with torch.no_grad():
            proba = model(input_tensor).item()

        if proba > 0.5:
            recommendations.append((course['name'], proba))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    print("\n🤖 Нейросеть рекомендует:")
    for name, prob in recommendations:
        print(f"{name} → Вероятность успеха: {prob:.2f}")

if __name__ == '__main__':
    # Загружаем модель и признаки
    model, features = load_model()

    # Пример студента
    student_example = {
        'gpa': 4.5,
        'year_of_study': 2,
        'major_CS': 1,
        'major_History': 0,
        'major_Math': 0,
        'major_Physics': 0,
    }

    # Загружаем курсы
    courses = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')

    # Рекомендации
    recommend_courses_for_student(pd.Series(student_example), courses, model, features)
