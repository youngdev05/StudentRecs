import torch
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.NeuralNetTrainer import Net

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

        # Весовые коэффициенты для системы рекомендаций
        self.weights = {
            'difficulty': {
                'Easy': 0.9,
                'Medium': 0.7,
                'Hard': 0.5
            },
            'major': {
                'CS': {'Computer Science': 1.2, 'Math': 1.1, 'Economics': 0.9},
                'Math': {'Math': 1.3, 'Physics': 1.2, 'Computer Science': 1.0},
                'History': {'History': 1.4, 'Philosophy': 1.1},
                'Physics': {'Physics': 1.3, 'Math': 1.2}
            },
            'year': {
                1: {'Easy': 1.2, 'Medium': 0.8, 'Hard': 0.5},
                2: {'Easy': 1.1, 'Medium': 1.0, 'Hard': 0.7},
                3: {'Easy': 1.0, 'Medium': 1.1, 'Hard': 0.9},
                4: {'Easy': 0.9, 'Medium': 1.2, 'Hard': 1.1}
            },
            'gpa': lambda gpa: min(1.5, 0.7 + gpa / 3.0)
        }

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

    def calculate_final_score(self, student: dict, course: dict, raw_prob: float) -> float:
        """Комплексный расчет итогового балла курса для студента"""
        # Получаем основную специальность студента
        major = next(k.split('_')[1] for k, v in student.items()
                     if k.startswith('major_') and v == 1)

        # Базовый расчет с учетом сложности
        base_score = raw_prob * self.weights['difficulty'].get(course['difficulty_level'], 1.0)

        # Учет специализации
        major_factor = self.weights['major'].get(major, {}).get(course['category'], 1.0)

        # Учет года обучения
        year_factor = self.weights['year'].get(student['year_of_study'], {}).get(course['difficulty_level'], 1.0)

        # Учет GPA
        gpa_factor = self.weights['gpa'](student['gpa'])

        # Итоговый балл
        final_score = base_score * major_factor * year_factor * gpa_factor

        # Гарантируем разумные границы
        return max(0.3, min(0.97, final_score))

    def prepare_student_data(self, student_data: dict) -> pd.DataFrame:
        required_fields = {'gpa', 'year_of_study'}
        if not required_fields.issubset(student_data.keys()):
            missing = required_fields - student_data.keys()
            raise ValueError(f"Отсутствуют обязательные поля: {missing}")

        for major in ['CS', 'Math', 'History', 'Physics']:
            if f'major_{major}' not in student_data:
                student_data[f'major_{major}'] = 0

        return pd.DataFrame([student_data], columns=self.feature_names).fillna(0)

    def recommend_courses(self, student_data: dict, courses_df: pd.DataFrame,
                          top_n: int = 5) -> list:
        """Улучшенная система рекомендаций"""
        if not all([self.model, self.scaler, self.feature_names]):
            raise RuntimeError("Модель не загружена")

        recommendations = []
        student_df = self.prepare_student_data(student_data)

        for _, course in courses_df.iterrows():
            # Подготовка входных данных
            input_data = student_df.copy()
            input_data[f'category_{course["category"]}'] = 1
            input_data[f'difficulty_level_{course["difficulty_level"]}'] = 1
            input_data['credits'] = course['credits']

            # Предсказание
            input_scaled = self.scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            with torch.no_grad():
                raw_prob = self.model(input_tensor).item()

            # Комплексный расчет
            final_score = self.calculate_final_score(student_data, course, raw_prob)

            recommendations.append({
                'course_id': course['course_id'],
                'course_name': course['name'],
                'category': course['category'],
                'difficulty': course['difficulty_level'],
                'raw_probability': raw_prob,
                'final_score': final_score,
                'match_description': self.get_match_description(student_data, course)
            })

        # Сортировка и выбор топ-N
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        return recommendations[:top_n]

    def get_match_description(self, student: dict, course: dict) -> str:
        """Генерирует текстовое описание соответствия"""
        major = next(k.split('_')[1] for k, v in student.items()
                     if k.startswith('major_') and v == 1)

        descriptions = {
            'major': {
                'CS': {
                    'Computer Science': "Отлично подходит для CS-специалистов",
                    'Math': "Математика важна для программирования",
                    'Economics': "Экономика для IT-менеджеров"
                },
                'Math': {
                    'Math': "Основная математическая подготовка",
                    'Physics': "Физика для прикладной математики",
                    'Computer Science': "Вычислительные методы"
                }
            },
            'difficulty': {
                'Easy': "Базовый уровень сложности",
                'Medium': "Средний уровень сложности",
                'Hard': "Продвинутый уровень сложности"
            }
        }

        major_desc = descriptions['major'].get(major, {}).get(course['category'],
                                                              "Курс не относится к вашей основной специализации")
        diff_desc = descriptions['difficulty'].get(course['difficulty_level'], "")

        return f"{major_desc}. {diff_desc}"

    def print_recommendations(self, recommendations: list):
        if not recommendations:
            print("\n🤷 Нет подходящих рекомендаций")
            return

        print("\n🎓 Лучшие рекомендации:")
        for i, course in enumerate(recommendations, 1):
            diff_color = {
                'Easy': '\033[92m',  # зеленый
                'Medium': '\033[93m',  # желтый
                'Hard': '\033[91m'  # красный
            }.get(course['difficulty'], '\033[0m')

            score = course['final_score']
            if score > 0.8:
                score_color = '\033[92m'
            elif score > 0.6:
                score_color = '\033[93m'
            else:
                score_color = '\033[91m'

            print(f"{i}. \033[1m{course['course_name']}\033[0m ({course['category']})")
            print(f"   {diff_color}Сложность: {course['difficulty']}\033[0m")
            print(f"   {score_color}Рекомендационный балл: {score:.2f}\033[0m")
            print(f"   \033[94m{course['match_description']}\033[0m")
            print("-" * 60)


if __name__ == '__main__':
    try:
        recommender = CourseRecommender('C:/Users/dimas/CsvGenerator/src/course_recommender_nn.pt')
        courses_df = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')

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
            print(f"\n{'=' * 70}")
            print(f"📋 Профиль студента #{i}:")
            print(
                f"• Специальность: \033[1m{next(k.split('_')[1] for k, v in student.items() if k.startswith('major_') and v == 1)}\033[0m")
            print(f"• GPA: \033[1m{student['gpa']}\033[0m")
            print(f"• Год обучения: \033[1m{student['year_of_study']}\033[0m")

            recommendations = recommender.recommend_courses(student, courses_df)
            recommender.print_recommendations(recommendations)

    except Exception as e:
        logger.error(f"Ошибка: {e}")