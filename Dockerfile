FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Speaker voiceprint model (26 MB). If the download fails here, web.py fetches it on first start.
ENV VOICEPRINT_MODEL=/app/models/voxceleb_resnet34_LM.onnx
RUN mkdir -p /app/models && python -c "import hashlib, urllib.request; \
url = 'https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx'; \
data = urllib.request.urlopen(url, timeout=300).read(); \
assert hashlib.sha256(data).hexdigest() == '7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068', 'checksum mismatch'; \
open('/app/models/voxceleb_resnet34_LM.onnx', 'wb').write(data)" \
    || echo "voiceprint model not downloaded at build time"

COPY *.py ./
COPY static/ ./static/

CMD ["python", "bot.py"]
