FROM articleclustering

COPY wsgi.py /
COPY api.py /
CMD ["gunicorn", "-b", "0.0.0.0:80", "wsgi:app"]