import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(data_path='data'):
    students = pd.read_csv('C:/Users/dimas/CsvGenerator/data/students.csv')
    courses = pd.read_csv('C:/Users/dimas/CsvGenerator/data/courses.csv')
    enrollments = pd.read_csv('C:/Users/dimas/CsvGenerator/data/enrollments.csv')
    return students, courses, enrollments

def prepare_data(students, courses, enrollments):
    # Объединяем все три таблицы
    df = pd.merge(enrollments, students, on='student_id')
    df = pd.merge(df, courses, on='course_id')

    # Создаем целевую переменную
    df['high_grade'] = (df['grade'] >= 4).astype(int)

    # Кодируем категориальные признаки
    df = pd.get_dummies(df, columns=['major', 'category', 'difficulty_level'])

    # Убираем ненужные колонки
    X = df.drop(columns=['high_grade', 'grade', 'student_id', 'course_id', 'name', 'semester'])
    y = df['high_grade']

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return model, X.columns.tolist()

def save_model(model, feature_names, filename='course_recommender.pkl'):
    joblib.dump((model, feature_names), filename)
    print(f"✅ Модель сохранена в '{filename}'")

if __name__ == '__main__':
    print("🔄 Загрузка данных...")
    students, courses, enrollments = load_data()

    print("🔗 Подготовка данных...")
    X, y = prepare_data(students, courses, enrollments)

    print("🧠 Обучение модели...")
    model, features = train_model(X, y)

    print("💾 Сохранение модели...")
    save_model(model, features)