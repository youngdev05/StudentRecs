import pandas as pd
import random
import os
import numpy as np
from collections import defaultdict

# Настройки
num_students = 300
num_courses = 15
min_grades_per_student = 4
max_grades_per_student = 8

# Профили курсов и студентов
course_list = [
    {"name": "Базы данных", "profile": "программирование", "difficulty": "Средний", "credits": 5},
    {"name": "Математический анализ", "profile": "математика", "difficulty": "Продвинутый", "credits": 6},
    {"name": "Основы программирования", "profile": "программирование", "difficulty": "Базовый", "credits": 4},
    {"name": "Алгоритмы и структуры данных", "profile": "программирование", "difficulty": "Продвинутый", "credits": 6},
    {"name": "Компьютерные сети", "profile": "сети", "difficulty": "Средний", "credits": 5},
    {"name": "Операционные системы", "profile": "системы", "difficulty": "Средний", "credits": 5},
    {"name": "Дискретная математика", "profile": "математика", "difficulty": "Средний", "credits": 5},
    {"name": "Теория вероятностей", "profile": "математика", "difficulty": "Средний", "credits": 5},
    {"name": "Машинное обучение", "profile": "AI", "difficulty": "Продвинутый", "credits": 6},
    {"name": "Веб-разработка", "profile": "программирование", "difficulty": "Средний", "credits": 5},
    {"name": "Архитектура ЭВМ", "profile": "железо", "difficulty": "Средний", "credits": 5},
    {"name": "Информационная безопасность", "profile": "сети", "difficulty": "Продвинутый", "credits": 6},
    {"name": "Системное программирование", "profile": "системы", "difficulty": "Продвинутый", "credits": 6},
    {"name": "Численные методы", "profile": "математика", "difficulty": "Средний", "credits": 5},
    {"name": "Компьютерная графика", "profile": "AI", "difficulty": "Средний", "credits": 5},
]

course_profiles = list(set(c["profile"] for c in course_list))
course_difficulties = ["Базовый", "Средний", "Продвинутый"]

student_profiles = [
    "программист", "аналитик", "сетевик", "data scientist", "системщик"
]

# Генерация студентов

def generate_students(n):
    data = []
    for i in range(1, n + 1):
        profile = random.choice(student_profiles)
        year = random.randint(1, 4)
        base_gpa = 65 + year * 4 + random.randint(-7, 7)
        gpa = int(np.clip(np.random.normal(base_gpa, 10), 40, 100))
        motivation = round(np.clip(np.random.beta(2, 2), 0, 1), 2)
        # Любимые и нелюбимые профили
        favorite_profile = random.choice(course_profiles)
        unfavorite_profile = random.choice([p for p in course_profiles if p != favorite_profile])
        data.append({
            'student_id': i,
            'student_profile': profile,
            'year_of_study': year,
            'gpa': gpa,
            'motivation': motivation,
            'favorite_profile': favorite_profile,
            'unfavorite_profile': unfavorite_profile
        })
    return pd.DataFrame(data)

students = generate_students(num_students)

# Генерация курсов
courses = pd.DataFrame([
    {
        'course_id': 100 + idx + 1,
        'name': c['name'],
        'course_profile': c['profile'],
        'course_difficulty': c['difficulty'],
        'course_credits': c['credits'],
        'individual_difficulty': np.random.uniform(-0.3, 0.3)
    }
    for idx, c in enumerate(course_list)
])

# Функция генерации оценок

def generate_grade(student, course):
    # Базовая вероятность успеха зависит от совпадения профиля и сложности
    gpa_norm = (student['gpa'] - 40) / 60
    motivation = student['motivation']
    profile_bonus = 0
    if course['course_profile'] == student['favorite_profile']:
        profile_bonus += 0.15
    if course['course_profile'] == student['unfavorite_profile']:
        profile_bonus -= 0.15
    if (
        (student['student_profile'] == "программист" and course['course_profile'] == "программирование") or
        (student['student_profile'] == "аналитик" and course['course_profile'] == "математика") or
        (student['student_profile'] == "сетевик" and course['course_profile'] == "сети") or
        (student['student_profile'] == "data scientist" and course['course_profile'] in ["AI", "математика"]) or
        (student['student_profile'] == "системщик" and course['course_profile'] in ["системы", "железо"])
    ):
        profile_bonus += 0.18
    # Сложность
    diff = {"Базовый": 0, "Средний": -0.18, "Продвинутый": -0.35}[course['course_difficulty']]
    # Индивидуальная сложность
    ind_diff = course['individual_difficulty']
    # Шум
    noise = np.random.normal(0, 0.18)
    score = 0.55 * gpa_norm + 0.25 * motivation + profile_bonus + diff + ind_diff + noise
    score = max(0, min(1, score))
    grade = int(np.clip(np.random.normal(score * 100, 10), 0, 100))
    return grade

# Генерация зачислений

enrollments = []
course_popularity = defaultdict(int)

for _, student in students.iterrows():
    # Студент выбирает больше курсов по любимому профилю
    fav_courses = courses[courses['course_profile'] == student['favorite_profile']]
    other_courses = courses[courses['course_profile'] != student['favorite_profile']]
    num_courses_taken = random.randint(min_grades_per_student, max_grades_per_student)
    n_fav = min(len(fav_courses), random.randint(1, num_courses_taken // 2 + 1))
    n_other = num_courses_taken - n_fav
    chosen = pd.concat([
        fav_courses.sample(n=n_fav, replace=False),
        other_courses.sample(n=n_other, replace=False)
    ])
    for _, course in chosen.iterrows():
        grade = generate_grade(student, course)
        semester = random.choice(['Fall 2023', 'Spring 2023'])
        is_favorite = int(course['course_profile'] == student['favorite_profile'])
        is_unfavorite = int(course['course_profile'] == student['unfavorite_profile'])
        enrollments.append((student['student_id'], course['course_id'], grade, semester, is_favorite, is_unfavorite))
        course_popularity[course['course_id']] += 1

enrollments = pd.DataFrame(enrollments, columns=['student_id', 'course_id', 'grade', 'semester', 'is_favorite', 'is_unfavorite'])

# Сохранение базовых CSV
os.makedirs("data", exist_ok=True)
students.to_csv('data/students.csv', index=False)
courses.to_csv('data/courses.csv', index=False)
enrollments.to_csv('data/enrollments.csv', index=False)

# Генерация student_course_success.csv
merged = enrollments.merge(students, on='student_id', how='left')
merged = merged.merge(courses, on='course_id', how='left')

def calculate_success(row):
    grade = row['grade']
    # Порог успеха зависит от сложности
    if row['course_difficulty'] == 'Базовый':
        if grade >= 60:
            return 1
        elif grade >= 45:
            return 0.5
        else:
            return 0
    if row['course_difficulty'] == 'Средний':
        if grade >= 65:
            return 1
        elif grade >= 50:
            return 0.5
        else:
            return 0
    if row['course_difficulty'] == 'Продвинутый':
        if row['gpa'] < 60:
            return 0
        if grade >= 70:
            return 1
        elif grade >= 55:
            return 0.5
        else:
            return 0

merged['success'] = merged.apply(calculate_success, axis=1)
success_df = merged[['student_id', 'course_id', 'success']]
success_df.to_csv('data/student_course_success.csv', index=False)

# Формируем обучающую выборку с новыми признаками
feature_df = merged[[
    'student_id', 'course_id', 'student_profile', 'year_of_study', 'gpa', 'motivation',
    'favorite_profile', 'unfavorite_profile', 'course_profile', 'course_difficulty', 'course_credits',
    'is_favorite', 'is_unfavorite', 'success'
]]
feature_df.to_csv('data/training_data.csv', index=False)
print("✅ Обучающая выборка training_data.csv сохранена.")

print("✅ Все данные успешно сгенерированы!")
print(f"Всего студентов: {num_students}")
print(f"Всего курсов: {num_courses}")
print(f"Всего записей о зачислениях: {len(enrollments)}")