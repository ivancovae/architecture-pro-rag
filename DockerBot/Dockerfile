FROM python:3.12-slim

# Базовые зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем requirements
COPY requirements.txt /app/requirements.txt

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . /app

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Запуск Telegram-бота
CMD ["python", "task4/bot.py"]
