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

        data.append({
            'student_id': i,
            'major': major,
            'year_of_study': year,
            'gpa': gpa
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


# Функция генерации оценок с учетом факторов
def generate_grade(student, course):
    # Базовые факторы
    difficulty_weight = {'Easy': 0, 'Medium': -0.3, 'Hard': -0.6}
    major_affinity = {
        'CS': {'Computer Science': 0.4, 'Math': 0.3, 'Economics': 0.2},
        'Math': {'Math': 0.5, 'Physics': 0.4, 'Computer Science': 0.3},
        'History': {'History': 0.6, 'Philosophy': 0.5},
        'Physics': {'Physics': 0.6, 'Math': 0.4}
    }

    # Расчет "идеальной" оценки
    base = student['gpa'] / 5.0
    diff = difficulty_weight[course['difficulty_level']]
    affinity = major_affinity[student['major']].get(course['category'], 0)
    year_bonus = student['year_of_study'] * 0.05

    # Итоговый score
    score = base + diff + affinity + year_bonus + random.uniform(-0.2, 0.2)
    score = max(0, min(1, score))

    # Преобразуем в 5-балльную шкалу
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
    diff = {'Easy': 0, 'Medium': -0.5, 'Hard': -1}[row['difficulty_level']]
    expected = (row['gpa'] / 5 * 2) + 2 + diff
    return 1 if grade >= expected else (0.5 if grade >= expected - 1 else 0)


merged['success'] = merged.apply(calculate_success, axis=1)
success_df = merged[['student_id', 'course_id', 'success']]

# Сохранение
success_df.to_csv('data/student_course_success.csv', index=False)

print("✅ Все данные успешно сгенерированы!")
print(f"Всего студентов: {num_students}")
print(f"Всего курсов: {num_courses}")
print(f"Всего записей о зачислениях: {len(enrollments)}")