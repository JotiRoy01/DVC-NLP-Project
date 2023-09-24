# featurization stage

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "zebra apple ball cat",
    "ball cat dog elphant",
    "very very unique"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())  
```