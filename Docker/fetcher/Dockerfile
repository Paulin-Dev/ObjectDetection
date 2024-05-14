FROM debian:latest

RUN apt update && apt install -y wget libimage-exiftool-perl

WORKDIR /app

ADD fetch.sh .

RUN chmod +x fetch.sh

CMD ["./fetch.sh"]
