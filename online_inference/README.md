Usage
-----
To **build** and **run** service on `http://0.0.0.0:8000/` use:
```
docker build -t heart_desease:v1 .
docker run -p 8000:8000 heart_desease:v1
```
To **pull** image from `https://hub.docker.com/` and **run** service use: 
```
docker pull artem09871/health_disease:v2
docker run -p 8000:8000 artem09871/health_disease:v2
```
Minimizing python docker images
-----
Благодаря использованию базового образа `python:3.7-slim-stretch` 
вместо `python:3.7` и параметра `--no-cache-dir` при установке
необходимых зависимостей удалось уменьшить размер docker image 
ровно в 3 раза (1.29 GB &#8594; 427 MB).