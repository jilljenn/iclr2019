all: venv tfidf.npz

venv:
	python -m venv venv
	. venv/bin/activate
	pip install -r requirements.txt

tfidf.npz: accepted.html
	python scrap.py

clean:
	rm -f tfidf.npz tfidf.txt
