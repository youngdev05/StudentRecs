import torch
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.NeuralNetTrainer import Net

# Разрешаем загрузку всех необходимых компонентов
torch.serialization.add_safe_globals([
    StandardScaler,
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.frombuffer
])

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
        """Загрузка обученной модели и scaler'а"""
        try:
            # Временное решение - отключаем weights_only для совместимости
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


    def prepare_student_data(self, student_data: dict) -> pd.DataFrame:
        """Подготовка данных студента для модели"""
        # Проверка обязательных полей
        required_fields = {'gpa', 'year_of_study'}
        if not required_fields.issubset(student_data.keys()):
            missing = required_fields - student_data.keys()
            raise ValueError(f"Отсутствуют обязательные поля: {missing}")

        # Добавление недостающих полей
        for major in ['CS', 'Math', 'History', 'Physics']:
            if f'major_{major}' not in student_data:
                student_data[f'major_{major}'] = 0

        return pd.DataFrame([student_data], columns=self.feature_names).fillna(0)

    def recommend_courses(self, student_data: dict, courses_df: pd.DataFrame, threshold: float = 0.5) -> list:
        """Рекомендация курсов для студента"""
        if not all([self.model, self.scaler, self.feature_names]):
            raise RuntimeError("Модель не загружена")

        recommendations = []
        student_df = self.prepare_student_data(student_data)

        for _, course in courses_df.iterrows():
            # Копируем и дополняем данные
            input_data = student_df.copy()
            input_data[f'category_{course["category"]}'] = 1
            input_data[f'difficulty_level_{course["difficulty_level"]}'] = 1
            input_data['credits'] = course['credits']

            # Нормализация и предсказание
            input_scaled = self.scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            with torch.no_grad():
                proba = self.model(input_tensor).item()

            if proba >= threshold:
                recommendations.append({
                    'course_id': course['course_id'],
                    'course_name': course['name'],
                    'category': course['category'],
                    'difficulty': course['difficulty_level'],
                    'success_probability': proba
                })

        # Сортировка по вероятности успеха
        return sorted(recommendations, key=lambda x: x['success_probability'], reverse=True)


# Пример использования
if __name__ == '__main__':
    try:
        # Инициализация рекомендателя
        recommender = CourseRecommender('C:/Users/dimas/CsvGenerator/src/course_recommender_nn.pt')

        # Пример данных студента
        student_data = {
            'gpa': 3.0,  # Попробуйте разные значения GPA
            'year_of_study': 3,  # Измените год обучения
            'major_CS': 0,  # Попробуйте другие специальности
            'major_Math': 1,
            'major_History': 0,
            'major_Physics': 0
        }

        # Загрузка данных о курсах
        courses_df = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')

        # Получение рекомендаций
        recommendations = recommender.recommend_courses(student_data, courses_df)

        # Вывод результатов
        print("\n🎓 Рекомендованные курсы:")
        for course in recommendations[:5]:
            print(
                f"{course['course_name']} ({course['category']}) - Вероятность успеха: {course['success_probability']:.2f}")

    except Exception as e:
        logger.error(f"Ошибка: {e}")
