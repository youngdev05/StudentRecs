import torch
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.NeuralNetTrainer import Net

# Разрешаем загрузку компонентов
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

        # Коэффициенты калибровки
        self.difficulty_weights = {
            'Easy': 0.85,
            'Medium': 0.75,
            'Hard': 0.65
        }
        self.major_weights = {
            'CS': {'Computer Science': 1.1, 'Math': 1.0, 'Economics': 0.9},
            'Math': {'Math': 1.2, 'Physics': 1.1, 'Computer Science': 1.0},
            'History': {'History': 1.3, 'Philosophy': 1.1},
            'Physics': {'Physics': 1.2, 'Math': 1.1}
        }

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Загрузка модели с дополнительной проверкой"""
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

    def calibrate_probability(self, raw_prob: float, student: dict, course: dict) -> float:
        """Расширенная калибровка вероятности"""
        # Базовое понижение уверенности
        calibrated = 0.5 + (raw_prob - 0.5) * 0.6

        # Учет сложности курса
        calibrated *= self.difficulty_weights.get(course['difficulty_level'], 1.0)

        # Учет специализации
        student_major = next(k.split('_')[1] for k, v in student.items()
                             if k.startswith('major_') and v == 1)
        major_factor = self.major_weights.get(student_major, {}).get(course['category'], 1.0)
        calibrated *= major_factor

        # Учет GPA (чем выше GPA, тем больше доверия)
        gpa_factor = min(1.0, student['gpa'] / 4.0)
        calibrated = calibrated * 0.8 + calibrated * gpa_factor * 0.2

        # Гарантированные границы
        return max(0.3, min(0.95, calibrated))

    def prepare_student_data(self, student_data: dict) -> pd.DataFrame:
        """Подготовка данных студента"""
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
        """Улучшенная система рекомендаций"""
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

            # Применяем расширенную калибровку
            proba = self.calibrate_probability(raw_proba, student_data, course)

            recommendations.append({
                'course_id': course['course_id'],
                'course_name': course['name'],
                'category': course['category'],
                'difficulty': course['difficulty_level'],
                'raw_probability': raw_proba,
                'calibrated_probability': proba,
                'match_score': self.calculate_match_score(student_data, course)
            })

        # Сортировка по комбинированному показателю
        sorted_recs = sorted(recommendations,
                             key=lambda x: (x['calibrated_probability'], x['match_score']),
                             reverse=True)

        # Фильтрация и ограничение количества
        filtered = [r for r in sorted_recs if r['calibrated_probability'] >= threshold]
        return filtered[:top_n]

    def calculate_match_score(self, student: dict, course: dict) -> float:
        """Дополнительная оценка соответствия курса студенту"""
        score = 0.0

        # Совпадение с основной специальностью
        student_major = next(k.split('_')[1] for k, v in student.items()
                             if k.startswith('major_') and v == 1)

        if student_major == 'CS' and course['category'] == 'Computer Science':
            score += 0.3
        elif student_major == 'Math' and course['category'] == 'Math':
            score += 0.4
        # ... другие правила соответствия

        # Учет года обучения
        year = student['year_of_study']
        if year >= 3 and course['difficulty_level'] == 'Hard':
            score += 0.2

        return min(1.0, score)

    def print_recommendations(self, recommendations: list):
        """Улучшенный вывод с цветовой маркировкой"""
        if not recommendations:
            print("\n🤷 Нет рекомендаций, соответствующих заданному порогу")
            return

        print("\n🎓 Рекомендованные курсы:")
        for i, course in enumerate(recommendations, 1):
            diff_color = {
                'Easy': '\033[92m',  # зеленый
                'Medium': '\033[93m',  # желтый
                'Hard': '\033[91m'  # красный
            }.get(course['difficulty'], '\033[0m')

            prob = course['calibrated_probability']
            if prob > 0.8:
                prob_color = '\033[92m'
            elif prob > 0.6:
                prob_color = '\033[93m'
            else:
                prob_color = '\033[91m'

            print(f"{i}. {course['course_name']} ({course['category']})")
            print(f"   {diff_color}Сложность: {course['difficulty']}\033[0m")
            print(f"   {prob_color}Вероятность успеха: {prob:.2f}\033[0m")
            print(f"   Совпадение: {course['match_score']:.2f}")
            print("-" * 50)


if __name__ == '__main__':
    try:
        recommender = CourseRecommender('C:/Users/dimas/CsvGenerator/src/course_recommender_nn.pt')
        courses_df = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')

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

        for i, student in enumerate(test_students, 1):
            print(f"\n{'=' * 60}")
            print(f"📊 Тестирование студента #{i}:")
            print(
                f"- Специальность: {next(k.split('_')[1] for k, v in student.items() if k.startswith('major_') and v == 1)}")
            print(f"- GPA: {student['gpa']}")
            print(f"- Год обучения: {student['year_of_study']}")

            recommendations = recommender.recommend_courses(
                student,
                courses_df,
                threshold=0.55,  # Повышенный порог
                top_n=5
            )
            recommender.print_recommendations(recommendations)

    except Exception as e:
        logger.error(f"Ошибка: {e}")