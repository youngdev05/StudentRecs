import torch
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.NeuralNetTrainer import Net

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseRecommender:
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, weights_only=False)
            self.model = Net(len(checkpoint['feature_names']))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.scaler = checkpoint['scaler']
            self.feature_names = checkpoint['feature_names']
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def recommend_courses(self, student_data: dict, courses_df: pd.DataFrame, top_n: int = 5) -> list:
        if not all([self.model, self.scaler, self.feature_names]):
            raise RuntimeError("Модель не загружена")

        print("\n--- DEBUG: FEATURE NAMES ---")
        print(self.feature_names)

        base_input = pd.DataFrame([np.zeros(len(self.feature_names))], columns=self.feature_names)

        for key, value in student_data.items():
            if key in base_input.columns:
                base_input.at[0, key] = value

        recommendations = []
        for _, course in courses_df.iterrows():
            input_data = base_input.copy()

            cat_col = f'course_category_{course["category"]}'
            diff_col = f'course_difficulty_{course["difficulty_level"]}'
            if cat_col in input_data.columns:
                input_data.at[0, cat_col] = 1
            if diff_col in input_data.columns:
                input_data.at[0, diff_col] = 1
            if 'credits' in input_data.columns:
                input_data.at[0, 'credits'] = course['credits']

            # Показываем входные данные до нормализации
            print(f"\n[{student_data['major']} | {course['name']}] RAW INPUT:")
            print(input_data)

            input_scaled = self.scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            with torch.no_grad():
                prob = self.model(input_tensor).item()

            # Показываем вероятность до калибровки
            print(f"[{student_data['major']} | {course['name']}] Prediction (before scale): {prob:.4f}")

            final_score = prob  # Используем "живую" вероятность без ограничений

            recommendations.append({
                'course': course['name'],
                'category': course['category'],
                'difficulty': course['difficulty_level'],
                'score': final_score
            })

        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]

    def print_recommendations(self, student: dict, recommendations: list):
        print(
            f"\nРекомендации для {student['major']} студента (GPA: {student['gpa']}, год: {student['year_of_study']}):")
        for i, course in enumerate(recommendations, 1):
            print(f"{i}. {course['course']} ({course['category']})")
            print(f"   Сложность: {course['difficulty']}")
            print(f"   Оценка: {course['score']:.2f}")
            print("-" * 40)


if __name__ == '__main__':
    try:
        recommender = CourseRecommender('C:/Users/dimas/CsvGenerator/src/course_recommender_nn.pt')
        courses_df = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')

        test_students = [
            {
                'gpa': 50,  # 100-балльная система
                'year_of_study': 3,
                'major_Math': 1,
                'major_CS': 0,
                'major_History': 0,
                'major_Physics': 0
            },
            {
                'gpa': 80,  # 100-балльная система
                'year_of_study': 4,
                'major_CS': 1,
                'major_Math': 0,
                'major_History': 0,
                'major_Physics': 0
            }
        ]

        for student in test_students:
            major = next(k.split('_')[1] for k, v in student.items() if k.startswith('major_') and v == 1)
            student['major'] = major
            recommendations = recommender.recommend_courses(student, courses_df)
            recommender.print_recommendations(student, recommendations)

    except Exception as e:
        logger.error(f"Ошибка: {e}")
