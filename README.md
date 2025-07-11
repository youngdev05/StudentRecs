# Рекомендательная система технических курсов

## Описание проекта

Данный проект — это рекомендательная система для студентов технических специальностей. Система генерирует синтетические данные о студентах и курсах, обучает нейросетевую модель и выдает индивидуальные рекомендации по выбору курсов с оценкой вероятности успеха.

**Ключевые особенности:**
- Только технические курсы (программирование, математика, AI, сети, системы и др.)
- Каждый студент имеет специализацию (например, "программист", "аналитик", "сетевик" и т.д.), любимый и нелюбимый профиль курсов
- Вероятность успеха зависит от совпадения профиля студента и курса, сложности, мотивации и других факторов
- Вся система работает в 100-балльной шкале (GPA, оценки, пороги успеха)
- Современный интерфейс на Streamlit

---

## Структура проекта

- **src/dataGenerator.py** — генерация студентов, курсов, зачислений и меток успеха. Формирует обучающую выборку с новыми признаками (профиль студента, любимый/нелюбимый профиль и др.)
- **src/NeuralNetTrainer.py** — обучение нейросетевой модели на сгенерированных данных
- **src/Predictor.py** — загрузка модели и выдача рекомендаций по курсам для заданного студента
- **src/app.py** — веб-интерфейс на Streamlit для ввода данных и получения рекомендаций
- **data/** — сгенерированные датасеты (students.csv, courses.csv, enrollments.csv, training_data.csv и др.)

---

## Как работает система

1. **Генерация данных**
   - Студенты получают специализацию, любимый и нелюбимый профиль курсов, мотивацию, GPA
   - Курсы имеют профиль (например, "программирование", "математика"), сложность (базовый/средний/продвинутый), кредиты
   - Вероятность успеха на курсе зависит от совпадения профилей, сложности, мотивации и индивидуальных особенностей
   - Все оценки и успехи в 100-балльной шкале

2. **Обучение модели**
   - Используется нейросеть на PyTorch
   - Входные признаки: профили, мотивация, GPA, любимый/нелюбимый профиль, сложность и др. (one-hot кодирование)
   - Модель обучается предсказывать вероятность успеха (0, 0.5, 1)

3. **Рекомендации**
   - Для заданного студента система рассчитывает вероятность успеха на каждом курсе
   - Выводит топ-5 курсов с наибольшей вероятностью успеха
   - В интерфейсе можно увидеть вероятности по всем курсам (график)

---

## Как запустить

1. Сгенерировать новые данные:
   ```bash
   python src/dataGenerator.py
   ```
2. Обучить модель:
   ```bash
   python src/NeuralNetTrainer.py
   ```
3. Запустить веб-интерфейс:
   ```bash
   streamlit run src/app.py
   ```

---
