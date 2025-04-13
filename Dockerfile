FROM stablebaselines/stable-baselines:latest

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

#TBA Start of application