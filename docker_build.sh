#!/usr/bin/env bash

# 1. Сборка Docker-образа на Ubuntu 24.04 с пакетами 2026 года
docker build -t yaroslawwinvasilev/meg_pipeline:2026.1 .

# 2. Авторизация в Docker Hub
docker login

# 3. Публикация образа
docker push yaroslawwinvasilev/meg_pipeline:2026.1
docker run --rm yaroslawwinvasilev/meg_pipeline:2026.1 pip freeze > requirements.txt

