import pandas as pd
import random
import os
import numpy as np
from collections import defaultdict

# Настройки
num_students = 300  # Увеличил для более репрезентативных данных
num_courses = 24  # Больше курсов для разнообразия
min_grades_per_student = 3
max_grades_per_student = 8  # Студенты могут брать больше курсов

# Категории курсов и их сложность
categories = {
    'Math': {'difficulty': 'Hard', 'credits': 6},
    'Computer Science': {'difficulty': 'Medium', 'credits': 6},
    'Physics': {'difficulty': 'Hard', 'credits': 6},
    'History': {'difficulty': 'Easy', 'credits': 4},
    'Philosophy': {'difficulty': 'Easy', 'credits': 4},
    'Economics': {'difficulty': 'Medium', 'credits': 4}
}


# Генерация студентов с более осмысленными характеристиками
def generate_students(n):
    data = []
    for i in range(1, n + 1):
        major = random.choice(['CS', 'Math', 'History', 'Physics'])
        year = random.randint(1, 4)

        # GPA зависит от специальности и года обучения
        base_gpa = {
                       'CS': 3.2, 'Math': 3.0, 'History': 3.5, 'Physics': 3.1
                   }[major] + (year * 0.1)

        gpa = max(2.0, min(5.0, round(np.random.normal(base_gpa, 0.5), 2)))

        # Новый признак: мотивация (от 0 до 1)
        motivation = round(np.clip(np.random.beta(2, 2), 0, 1), 2)

        data.append({
            'student_id': i,
            'major': major,
            'year_of_study': year,
            'gpa': gpa,
            'motivation': motivation
        })
    return pd.DataFrame(data)


students = generate_students(num_students)

# Генерация курсов с зависимостью от категории
courses = pd.DataFrame({
    'course_id': range(101, 101 + num_courses),
    'name': [f'Course {i}' for i in range(1, num_courses + 1)],
    'category': [random.choice(list(categories.keys())) for _ in range(num_courses)]
})
courses['difficulty_level'] = courses['category'].map(lambda x: categories[x]['difficulty'])
courses['credits'] = courses['category'].map(lambda x: categories[x]['credits'])
# Добавляем индивидуальную сложность
courses['individual_difficulty'] = np.random.uniform(-0.15, 0.15, size=len(courses))


# Функция генерации оценок с учетом факторов
def generate_grade(student, course):
    # Жёсткие правила для генерации успеха
    if course['difficulty_level'] == 'Hard' and student['gpa'] < 2.7:
        score = 0.3 + np.random.normal(0, 0.05)  # всегда низко
    elif course['difficulty_level'] == 'Easy' and student['gpa'] > 4.0:
        score = 0.9 + np.random.normal(0, 0.05)  # всегда высоко
    else:
        difficulty_weight = {'Easy': 0, 'Medium': -0.25, 'Hard': -0.5}
        major_affinity = {
            'CS': {'Computer Science': 0.2, 'Math': 0.1, 'Economics': 0.05},
            'Math': {'Math': 0.25, 'Physics': 0.15, 'Computer Science': 0.1},
            'History': {'History': 0.3, 'Philosophy': 0.2},
            'Physics': {'Physics': 0.3, 'Math': 0.15}
        }
        motivation = student.get('motivation', 0.5)
        motivation_effect = (motivation - 0.5) * 0.15
        gpa_norm = (student['gpa'] - 2.0) / 3.0
        gpa_norm = max(0, min(1, gpa_norm))
        diff = difficulty_weight[course['difficulty_level']]
        affinity = major_affinity[student['major']].get(course['category'], 0)
        year_bonus = student['year_of_study'] * 0.02
        noise = np.random.normal(0, 0.10)
        individual_diff = course.get('individual_difficulty', 0)
        if course['difficulty_level'] == 'Easy':
            score = 0.55 * gpa_norm + 0.15 + affinity + year_bonus + motivation_effect + noise + individual_diff
        elif course['difficulty_level'] == 'Medium':
            score = 0.45 * gpa_norm + diff + affinity + year_bonus + motivation_effect + noise + individual_diff
        else:  # Hard
            score = 0.35 * gpa_norm + diff + affinity + year_bonus + motivation_effect + noise + individual_diff
        if student['gpa'] < 2.5:
            if course['difficulty_level'] == 'Hard':
                score = min(score, 0.45)
            elif course['difficulty_level'] == 'Medium':
                score = min(score, 0.6)
        if student['gpa'] > 4.0:
            if course['difficulty_level'] == 'Easy':
                score = max(score, 0.8)
            elif course['difficulty_level'] == 'Medium':
                score = max(score, 0.7)
            elif course['difficulty_level'] == 'Hard':
                score = max(score, 0.6)
    score = max(0, min(1, score))
    if score < 0.6:
        return random.choice([2, 2.5, 3])
    elif score < 0.75:
        return random.choice([3, 3.5])
    elif score < 0.9:
        return random.choice([3.5, 4])
    else:
        return random.choice([4, 4.5, 5])


# Генерация зачислений с осмысленными оценками
enrollments = []
course_popularity = defaultdict(int)

for _, student in students.iterrows():
    # Студенты выбирают курсы не совсем случайно
    preferred_categories = {
        'CS': ['Computer Science', 'Math', 'Economics'],
        'Math': ['Math', 'Physics', 'Computer Science'],
        'History': ['History', 'Philosophy'],
        'Physics': ['Physics', 'Math']
    }[student['major']]

    # Выбираем курсы с приоритетом к своей специальности
    available_courses = courses[courses['category'].isin(preferred_categories + ['Economics'])]
    num_courses_taken = random.randint(min_grades_per_student, max_grades_per_student)

    # Выбор курсов с учетом популярности (чтобы были более и менее популярные курсы)
    chosen = available_courses.sample(
        min(len(available_courses), num_courses_taken),
        weights=1 / (1 + available_courses['course_id'].map(course_popularity))
    )

    for _, course in chosen.iterrows():
        grade = generate_grade(student, course)
        semester = random.choice(['Fall 2023', 'Spring 2023'])
        enrollments.append((student['student_id'], course['course_id'], grade, semester))
        course_popularity[course['course_id']] += 1

enrollments = pd.DataFrame(enrollments, columns=['student_id', 'course_id', 'grade', 'semester'])

# Сохранение базовых CSV
os.makedirs("data", exist_ok=True)
students.to_csv('data/students.csv', index=False)
courses.to_csv('data/courses.csv', index=False)
enrollments.to_csv('data/enrollments.csv', index=False)

# Генерация student_course_success.csv с более тонкой градацией
merged = enrollments.merge(students, on='student_id', how='left')
merged = merged.merge(courses, on='course_id', how='left')


# Успешность теперь не бинарная, а с градациями
def calculate_success(row):
    grade = row['grade']
    # Жёсткое правило: слабый студент не может быть успешен на сложном курсе
    if row['difficulty_level'] == 'Hard' and row['gpa'] < 2.7:
        return 0
    # Сильный студент на лёгком курсе — успех только если оценка высокая
    if row['difficulty_level'] == 'Easy' and row['gpa'] > 4.0:
        return 1 if grade >= 4 else 0
    # Для остальных успех только если grade >= 3.5
    return 1 if grade >= 3.5 else 0


merged['success'] = merged.apply(calculate_success, axis=1)
success_df = merged[['student_id', 'course_id', 'success']]

# Сохранение
success_df.to_csv('data/student_course_success.csv', index=False)

# 🔧 Формируем обучающую выборку X с нужными фичами
feature_df = merged[[
    'student_id', 'course_id', 'major', 'year_of_study', 'gpa', 'motivation',
    'category', 'difficulty_level', 'credits', 'success'
]]
feature_df.rename(columns={
    'category': 'course_category',
    'difficulty_level': 'course_difficulty',
    'credits': 'course_credits'
}, inplace=True)

feature_df.to_csv('data/training_data.csv', index=False)
print("✅ Обучающая выборка training_data.csv сохранена.")


print("✅ Все данные успешно сгенерированы!")
print(f"Всего студентов: {num_students}")
print(f"Всего курсов: {num_courses}")
print(f"Всего записей о зачислениях: {len(enrollments)}")