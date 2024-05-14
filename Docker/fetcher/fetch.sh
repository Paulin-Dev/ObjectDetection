#!/bin/bash


for _ in $(seq 1 "$LOOP"); do

    filename="/tmp/CWL9018.jpg"

    wget -O "$filename" "$URL"

    fileSize=$(exiftool -s -s -s -n -FileSize "$filename")

    if ((fileSize > 10000)); then
        createDate=$(exiftool -s -s -s -d "%Y%m%d%H%M%S" -CreateDate "$filename")
        mv -n "$filename" "/images/$createDate.jpg"
    fi

    rm -f "$filename"
    sleep "$SLEEP_TIME"

done
