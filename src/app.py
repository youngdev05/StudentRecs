import streamlit as st
import pandas as pd
import os
from Predictor import CourseRecommender

st.set_page_config(page_title="Рекомендательная система курсов", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "course_recommender_nn.pt")
COURSES_PATH = os.path.join(os.path.dirname(__file__), "../data/courses.csv")

# Загрузка модели и курсов
@st.cache_resource
def load_recommender():
    recommender = CourseRecommender(MODEL_PATH)
    return recommender

@st.cache_data
def load_courses():
    return pd.read_csv(COURSES_PATH)

recommender = load_recommender()
courses_df = load_courses()

st.title("🎓 Рекомендательная система курсов")
st.write("Введите свои данные, чтобы получить рекомендации по курсам и вероятность успеха!")

student_profiles = ["программист", "аналитик", "сетевик", "data scientist", "системщик"]
course_profiles = ["программирование", "математика", "сети", "AI", "системы", "железо"]

student_profile = st.selectbox("Ваш профиль (специализация)", student_profiles)
gpa = st.slider("Ваш GPA (от 40 до 100)", 40, 100, 70)
motivation = st.slider("Мотивация (от 0 до 1)", 0.0, 1.0, 0.5, step=0.01)
year_of_study = st.selectbox("Курс обучения", [1, 2, 3, 4])
favorite_profile = st.selectbox("Любимый профиль курсов", course_profiles)
unfavorite_profile = st.selectbox("Нелюбимый профиль курсов", [p for p in course_profiles if p != favorite_profile])

student_data = {
    "gpa": gpa,
    "year_of_study": year_of_study,
    "motivation": motivation,
    "student_profile": student_profile,
    "favorite_profile": favorite_profile,
    "unfavorite_profile": unfavorite_profile
}

if st.button("Показать рекомендации!"):
    recommendations = recommender.recommend_courses(student_data, courses_df, top_n=5)
    st.subheader("Топ-5 курсов для вас:")
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}. {rec['course']}**  ")
        st.write(f"Профиль: {rec['profile']}, Сложность: {rec['difficulty']}, Вероятность успеха: {rec['score']:.1%}")
    # Визуализация вероятностей по всем курсам
    st.subheader("Ваши шансы на всех курсах:")
    all_recs = recommender.recommend_courses(student_data, courses_df, top_n=len(courses_df))
    chart_df = pd.DataFrame({
        "Курс": [r["course"] for r in all_recs],
        "Вероятность успеха": [r["score"] for r in all_recs]
    })
    st.bar_chart(chart_df.set_index("Курс")) 