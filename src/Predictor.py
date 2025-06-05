import torch
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
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
        self.calibration_factor = 0.7  # Коэффициент калибровки уверенности

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Загрузка обученной модели и scaler'а"""
        try:
            checkpoint = torch.load(model_path, weights_only=False)
            self.model = Net(len(checkpoint['feature_names']))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.scaler = checkpoint['scaler']
            self.feature_names = checkpoint['feature_names']

            # Автоподбор коэффициента калибровки на основе сложности курсов
            if 'course_difficulty_weights' in checkpoint:
                self.difficulty_weights = checkpoint['course_difficulty_weights']
            else:
                self.difficulty_weights = {'Easy': 1.0, 'Medium': 0.9, 'Hard': 0.8}

            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def calibrate_probability(self, raw_prob: float, difficulty: str) -> float:
        """Калибровка вероятности с учетом сложности курса"""
        # Базовая калибровка
        calibrated = 0.5 + (raw_prob - 0.5) * self.calibration_factor

        # Учет сложности курса
        calibrated *= self.difficulty_weights.get(difficulty, 1.0)

        # Гарантируем границы [0.05, 0.95]
        return max(0.05, min(0.95, calibrated))

    def prepare_student_data(self, student_data: dict) -> pd.DataFrame:
        """Подготовка данных студента для модели"""
        required_fields = {'gpa', 'year_of_study'}
        if not required_fields.issubset(student_data.keys()):
            missing = required_fields - student_data.keys()
            raise ValueError(f"Отсутствуют обязательные поля: {missing}")

        for major in ['CS', 'Math', 'History', 'Physics']:
            if f'major_{major}' not in student_data:
                student_data[f'major_{major}'] = 0

        return pd.DataFrame([student_data], columns=self.feature_names).fillna(0)

    def recommend_courses(self, student_data: dict, courses_df: pd.DataFrame,
                          threshold: float = 0.5, top_n: int = 5) -> list:
        """Рекомендация курсов для студента с калибровкой"""
        if not all([self.model, self.scaler, self.feature_names]):
            raise RuntimeError("Модель не загружена")

        recommendations = []
        student_df = self.prepare_student_data(student_data)

        for _, course in courses_df.iterrows():
            input_data = student_df.copy()
            input_data[f'category_{course["category"]}'] = 1
            input_data[f'difficulty_level_{course["difficulty_level"]}'] = 1
            input_data['credits'] = course['credits']

            input_scaled = self.scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            with torch.no_grad():
                raw_proba = self.model(input_tensor).item()

            # Применяем калибровку
            proba = self.calibrate_probability(raw_proba, course['difficulty_level'])

            recommendations.append({
                'course_id': course['course_id'],
                'course_name': course['name'],
                'category': course['category'],
                'difficulty': course['difficulty_level'],
                'raw_probability': raw_proba,
                'calibrated_probability': proba
            })

        # Фильтрация и сортировка
        filtered = [r for r in recommendations if r['calibrated_probability'] >= threshold]
        sorted_recommendations = sorted(filtered,
                                        key=lambda x: x['calibrated_probability'],
                                        reverse=True)

        return sorted_recommendations[:top_n]

    def print_recommendations(self, recommendations: list):
        """Красивый вывод рекомендаций"""
        if not recommendations:
            print("\n🤷 Нет рекомендаций, соответствующих заданному порогу")
            return

        print("\n🎓 Рекомендованные курсы (вероятность успеха):")
        for i, course in enumerate(recommendations, 1):
            diff_emoji = {
                'Easy': '⭐',
                'Medium': '⭐⭐',
                'Hard': '⭐⭐⭐'
            }.get(course['difficulty'], '')

            print(f"{i}. {course['course_name']} ({course['category']}) {diff_emoji}")
            print(f"   Сложность: {course['difficulty']}")
            print(f"   Вероятность успеха: {course['calibrated_probability']:.2f}")
            print(f"   (Исходная: {course['raw_probability']:.2f})")
            print("-" * 40)


# Пример использования
if __name__ == '__main__':
    try:
        recommender = CourseRecommender('C:/Users/dimas/CsvGenerator/src/course_recommender_nn.pt')

        # Тестовые студенты
        test_students = [
            {
                'gpa': 3.0,
                'year_of_study': 3,
                'major_Math': 1,
                'major_CS': 0,
                'major_History': 0,
                'major_Physics': 0
            },
            {
                'gpa': 4.5,
                'year_of_study': 4,
                'major_CS': 1,
                'major_Math': 0,
                'major_History': 0,
                'major_Physics': 0
            }
        ]

        courses_df = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')

        for i, student in enumerate(test_students, 1):
            print(f"\n{'=' * 50}")
            print(f"📊 Тестирование студента #{i}:")
            print(f"- Специальность: {[k for k, v in student.items() if 'major_' in k and v == 1][0].split('_')[1]}")
            print(f"- GPA: {student['gpa']}")
            print(f"- Год обучения: {student['year_of_study']}")

            recommendations = recommender.recommend_courses(
                student,
                courses_df,
                threshold=0.4,  # Более низкий порог для демонстрации
                top_n=5
            )
            recommender.print_recommendations(recommendations)

    except Exception as e:
        logger.error(f"Ошибка: {e}")