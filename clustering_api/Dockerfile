FROM python:3.8

ENV REDIS=redis
COPY requirements.txt /
RUN pip install -r requirements.txt
# COPY cc.ta.100.bin /
# COPY cc.en.100.bin /
# CMD ["python", "wsgi.py"]

COPY wsgi.py /
COPY api.py /
CMD ["gunicorn", "-b", "0.0.0.0:80", "--timeout=1000", "wsgi:app"]