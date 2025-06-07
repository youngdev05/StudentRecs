import torch
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from NeuralNetTrainer import Net

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

        base_input = pd.DataFrame([np.zeros(len(self.feature_names))], columns=self.feature_names)

        # Заполняем признаки студента (числовые и one-hot)
        for key, value in student_data.items():
            if key in base_input.columns:
                base_input.at[0, key] = value
        # one-hot для student_profile, favorite_profile, unfavorite_profile
        for prefix in ["student_profile_", "favorite_profile_", "unfavorite_profile_"]:
            for col in base_input.columns:
                if col.startswith(prefix):
                    base_input.at[0, col] = 1 if col == f"{prefix}{student_data[prefix[:-1]]}" else 0

        recommendations = []
        for _, course in courses_df.iterrows():
            input_data = base_input.copy()
            # one-hot для course_profile, course_difficulty
            for prefix, value in [("course_profile_", course["course_profile"]), ("course_difficulty_", course["course_difficulty"])]:
                for col in input_data.columns:
                    if col.startswith(prefix):
                        input_data.at[0, col] = 1 if col == f"{prefix}{value}" else 0
            # credits
            if 'course_credits' in input_data.columns:
                input_data.at[0, 'course_credits'] = course['course_credits']
            # is_favorite/is_unfavorite
            if 'is_favorite' in input_data.columns:
                input_data.at[0, 'is_favorite'] = int(course['course_profile'] == student_data['favorite_profile'])
            if 'is_unfavorite' in input_data.columns:
                input_data.at[0, 'is_unfavorite'] = int(course['course_profile'] == student_data['unfavorite_profile'])

            input_scaled = self.scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            with torch.no_grad():
                prob = self.model(input_tensor).item()

            recommendations.append({
                'course': course['name'],
                'profile': course['course_profile'],
                'difficulty': course['course_difficulty'],
                'score': prob
            })

        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]

    def print_recommendations(self, student: dict, recommendations: list):
        print(
            f"\nРекомендации для {student['student_profile']} студента (GPA: {student['gpa']}, год: {student['year_of_study']}):")
        for i, course in enumerate(recommendations, 1):
            print(f"{i}. {course['course']} (Профиль: {course['profile']})")
            print(f"   Сложность: {course['difficulty']}")
            print(f"   Оценка: {course['score']:.2f}")
            print("-" * 40)


if __name__ == '__main__':
    try:
        recommender = CourseRecommender('C:/Users/dimas/CsvGenerator/src/course_recommender_nn.pt')
        courses_df = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')

        test_students = [
            {
                'gpa': 75,
                'year_of_study': 3,
                'motivation': 0.7,
                'student_profile': 'программист',
                'favorite_profile': 'программирование',
                'unfavorite_profile': 'математика'
            },
            {
                'gpa': 60,
                'year_of_study': 2,
                'motivation': 0.5,
                'student_profile': 'аналитик',
                'favorite_profile': 'математика',
                'unfavorite_profile': 'AI'
            },
            {
                'gpa': 85,
                'year_of_study': 4,
                'motivation': 0.9,
                'student_profile': 'сетевик',
                'favorite_profile': 'сети',
                'unfavorite_profile': 'железо'
            }
        ]

        for student in test_students:
            recommendations = recommender.recommend_courses(student, courses_df)
            recommender.print_recommendations(student, recommendations)

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
