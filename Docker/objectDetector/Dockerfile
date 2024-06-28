FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
