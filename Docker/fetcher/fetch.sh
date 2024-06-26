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

    # Check if the file size is bigger than 10ko
    fileSize=$(exiftool -s -s -s -n -FileSize "$filename")
    if [ "$fileSize" -gt 10000 ]; then

        createDate=$(exiftool -s -s -s -d "%Y%m%d%H%M%S" -CreateDate "$filename")

        # Check if file already exists (=same image), if so, retry 10s later
        if [ -f "/images/$createDate.jpg" ]; then
            retry
        else
            mv -n "$filename" "/images/$createDate.jpg"
        fi

    else
        retry
    fi
}


if [ "$LOOP" -eq -1 ]; then
    while true; do
        download
        rm -f "$filename"
        sleep "$SLEEP_TIME"
    done
else
    for _ in $(seq 1 "$LOOP"); do
        download
        rm -f "$filename"
        sleep "$SLEEP_TIME"
    done
fi

