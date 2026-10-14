# Trained model

`sms_spam_model.joblib` is the complete fitted pipeline: raw-message preprocessing,
TF-IDF, numeric scaling and SVM. Keep `sms_classifier.py` alongside the project so
joblib can import the custom transformer. NLTK stopwords and WordNet are installed
on first use by `setup_nltk()`.

The included model was created with scikit-learn 1.8.0. Use the pinned
`requirements.txt`, or retrain after changing scikit-learn versions. Only load
joblib files from trusted sources: deserialization can execute Python code.

Regenerate with `python train.py` from the project folder.
