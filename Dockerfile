FROM python:3.9-slim
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000

CMD [ "python3", "api.py"]
