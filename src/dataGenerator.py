import pandas as pd
import random

# Настройки
num_students = 30
num_courses = 12
min_grades_per_student = 3
max_grades_per_student = 5

# Категории курсов
categories = ['Math', 'History', 'Computer Science', 'Physics', 'Philosophy', 'Economics']
difficulties = ['Easy', 'Medium', 'Hard']

# Генерация студентов
students = pd.DataFrame({
    'student_id': range(1, num_students + 1),
    'major': [random.choice(['CS', 'Math', 'History', 'Physics']) for _ in range(num_students)],
    'year_of_study': [random.randint(1, 4) for _ in range(num_students)],
    'gpa': [round(random.uniform(2.0, 5.0), 2) for _ in range(num_students)]
})

# Генерация курсов
courses = pd.DataFrame({
    'course_id': range(101, 101 + num_courses),
    'name': [f'Course {i}' for i in range(1, num_courses + 1)],
    'category': [random.choice(categories) for _ in range(num_courses)],
    'difficulty_level': [random.choice(difficulties) for _ in range(num_courses)],
    'credits': [random.choice([4, 6]) for _ in range(num_courses)]
})

# Генерация зачислений
enrollments = []
for student_id in students['student_id']:
    num_courses_taken = random.randint(min_grades_per_student, max_grades_per_student)
    course_ids = random.sample(list(courses['course_id']), num_courses_taken)
    for cid in course_ids:
        grade = random.choice([2, 3, 3.5, 4, 4.5, 5])
        semester = random.choice(['Fall 2023', 'Spring 2023'])
        enrollments.append((student_id, cid, grade, semester))

enrollments = pd.DataFrame(enrollments, columns=['student_id', 'course_id', 'grade', 'semester'])

# Сохранение в CSV
students.to_csv('students.csv', index=False)
courses.to_csv('courses.csv', index=False)
enrollments.to_csv('enrollments.csv', index=False)

print("✅ Данные успешно сгенерированы!")