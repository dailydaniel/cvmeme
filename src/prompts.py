scale = """
1. Опыт работы:
   - 1: Менее года
   - 2: 1-2 года
   - 3: 3-5 лет
   - 4: 6-9 лет
   - 5: 10+ лет

2. Ответственность:
   - 1: Очень низкая (нельзя поручать задачи)
   - 2: Низкая (нужен сильный контроль)
   - 3: Средняя (нужно периодически проверять)
   - 4: Высокая (иногда лучше проверять)
   - 5: Очень высокая (можно положиться)

3. Креативность:
   - 1: Очень низкая (следует инструкциям)
   - 2: Низкая (вносит небольшие предложения)
   - 3: Средняя (иногда генерирует новые идеи)
   - 4: Высокая (часто предлагает инновационные решения)
   - 5: Очень высокая (постоянно предлагает инновационные решения)

4. Коммуникабельность:
   - 1: Очень низкая (затруднительное общение)
   - 2: Низкая (ограниченное общение)
   - 3: Средняя (нормальное общение)
   - 4: Высокая (хорошее общение)
   - 5: Очень высокая (исключительные навыки общения)

5. Лидерские качества:
   - 1: Очень низкие (никогда не берет на себя лидерство)
   - 2: Низкие (редко берет на себя лидерство)
   - 3: Средние (иногда берет на себя лидерство)
   - 4: Высокие (часто берет на себя лидерство)
   - 5: Очень высокие (постоянно проявляет лидерство)

6. Стрессоустойчивость:
   - 1: Очень низкая (не справляется со стрессом)
   - 2: Низкая (с трудом справляется со стрессом)
   - 3: Средняя (есть проблемы со стрессом)
   - 4: Высокая (нет проблем со стрессом)
   - 5: Очень высокая (исключительно справляется со стрессом)

7. Уровень сложности работы:
   - 1: Очень низкий (простая работа, не требующая навыков)
   - 2: Низкий (работа с базовыми требованиями)
   - 3: Средний (требует определенных навыков и знаний)
   - 4: Высокий (требует значительных знаний и навыков)
   - 5: Очень высокий (экспертный уровень, сложные задачи)

8. Средняя продолжительность работы на одном месте:
   - 1: Очень низкая (менее 6 месяцев)
   - 2: Низкая (6 месяцев - 1 год)
   - 3: Средняя (1-2 года)
   - 4: Высокая (2-4 лет)
   - 5: Очень высокая (более 4 лет)

9. Карьерный рост:
   - 1: Нет роста (остается на одном уровне)
   - 2: Низкий (незначительное повышение)
   - 3: Средний (умеренное повышение)
   - 4: Высокий (значительное повышение)
   - 5: Очень высокий (быстрое продвижение по карьерной лестнице)

10. Уровень дотошности описания:
    - 1: Очень низкий (очень мало деталей и коротко)
    - 2: Низкий (некоторые детали, но все равно коротко)
    - 3: Средний (достаточно подробное, но без лишних деталей)
    - 4: Высокий (подробное и хорошо описанное)
    - 5: Очень высокий (крайне детализированное описание)

11. Уровень заумности:
    - 1: Очень низкий (очень простые слова с ошибками)
    - 2: Низкий (простые слова без ошибок)
    - 3: Средний (обычные слова)
    - 4: Высокий (сложные слова)
    - 5: Очень высокий (очень сложные и редкие слова)

12. Уровень высокомерия:
    - 1: Очень низкий (очень скромный)
    - 2: Низкий (скромный)
    - 3: Средний (немного уверенный, но без высокомерия)
    - 4: Высокий (очень уверенный, с элементами высокомерия)
    - 5: Очень высокий (очень высокомерный)
    
13. Образование:
    - 1: Начальное, среднее или нет
    - 2: Высшее неоконченное или среднее специальное
    - 3: Высшее бакалавриат
    - 4: Высшее магистратура
    - 5: Высшее аспирантура или phd
    
14. Акадимичность:
    - 1: Только опыт работы
    - 2: Опыт работы свои проекты
    - 3: Опыт работы, свои проекты, диплом по специальности
    - 4: Свои проекты, диплом по специальности, публикации
    - 5: Диплом по специальности, публикации, преподавание
    
15. Иностранные языки:
    - 1: Нет
    - 2: Базовый английский
    - 3: Хороший английский
    - 4: Продвинутый английский или несколько языков
    - 5: Резюме на англияском
    
16. Профессиональные сертификаты:
    - 1: Нет
    - 2: Есть курс
    - 3: Есть несколько курсов
    - 4: Много курсов или расписанных сертификатов
    - 5: Много сертификатов и курсов
"""

prompt_rerank = """
Тебе дан текст резюме и три мема, для каждого мема дано название, короткое описание и описание каким бы мем был работником. 
Твоя задача - выбрать наиболее подходящий мем.

Текст резюме:
{cv}

Мем: {meme1_name}
Короткое описание: {meme1_desc}
Описание каким мем был бы работником: {meme1_worker}

Мем: {meme2_name}
Короткое описание: {meme2_desc}
Описание каким мем был бы работником: {meme2_worker}

Мем: {meme3_name}
Короткое описание: {meme3_desc}
Описание каким мем был бы работником: {meme3_worker}

Твой ответ ДОЛЖЕН содержать ТОЛЬКО название мема, больше никакого текста.
"""

cv_scale = "Тебе дан текст резюме и параметры оценки со шкалой. Твоя задача - используя параметры со шкалой, и текст резбме оценить это резюме.\nТекст резюме:\n{cv}\nПараметры со шкалой оценки:\n{scale}Ответ ДОЛЖЕН быть в формате json."

base_prompt = """
Ты бот, который подбирает мем по резюме. 
У нас есть несколько десятков тщательно отобраных мемов, которые мы соотносим с резюме.
Ответь на вопрос пользователя, если он на эту тему и попроси отправить pdf файл с резюме. 
Общайся неформально, на 'ты'.
НЕ ОТВЕЧАЙ на вопросы, которые не связаны с резюме.
Если тебя спросят, то мы не храним резюме пользователей.
"""
