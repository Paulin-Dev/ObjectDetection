#!/bin/bash


filename="/tmp/CWL9018.jpg"
retried=0


retry() {
    if [ "$retried" -eq 1 ]; then
        retried=0
    else
        retried=1
        sleep 10
        download
    fi
}


download () {
    local fileSize createDate

    wget -O "$filename" "$URL"

    # Check if the file size if bigger than 10ko
    fileSize=$(exiftool -s -s -s -n -FileSize "$filename")
    if [ "$fileSize" -gt 10000 ]; then
        createDate=$(exiftool -s -s -s -d "%Y%m%d%H%M%S" -CreateDate "$filename")
        mv -n "$filename" "/images/$createDate.jpg"
    else
        retry
    fi
}


for _ in $(seq 1 "$LOOP"); do

    download
    rm -f "$filename"
    sleep "$SLEEP_TIME"

done
