FROM python:3.8
COPY requirements.txt /
RUN pip install -r requirements.txt
COPY cc.ta.300.bin /
COPY cc.en.300.bin /
# CMD ["python", "wsgi.py"]

COPY wsgi.py /
COPY api.py /
CMD ["gunicorn", "-b", "0.0.0.0:80", "wsgi:app"]