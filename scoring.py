# scoring.py
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Lightweight skill list. Expand as needed.
SKILL_LIST = {
    'python', 'pandas', 'numpy', 'scikit-learn', 'sklearn', 'tensorflow',
    'pytorch', 'spacy', 'nlp', 'sql', 'spark', 'hadoop', 'aws', 'azure',
    'docker', 'kubernetes', 'git', 'tableau', 'power bi', 'xgboost',
    'lightgbm', 'catboost', 'random forest', 'gradient boosting',
    'logistic regression', 'svm', 'knn', 'decision tree', 'eda', 'feature engineering'
}

WEIGHTS = {
    'tfidf': 0.60,
    'keyword_cov': 0.25,
    'skill_bonus': 0.15
}

def load_nlp():
    import spacy
    return spacy.load('en_core_web_sm')

def build_vectorizer(corpus_texts: List[str]) -> TfidfVectorizer:
    # Keep it simple and robust; bi-grams help catch phrases
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1
    )
    vectorizer.fit(corpus_texts)
    return vectorizer

def _cosine_similarity(vec_a, vec_b):
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) or 1e-9
    return float(np.dot(vec_a, vec_b) / denom)

def _lemmatized_tokens(nlp, text: str) -> List[str]:
    doc = nlp(text)
    return [t.lemma_.lower().strip() for t in doc if t.is_alpha and not t.is_stop]

def _extract_keywords(nlp, jd_text: str) -> List[str]:
    # Nouns and verbs are decent approximations of “requirements”
    doc = nlp(jd_text)
    kws = []
    for t in doc:
        if t.pos_ in {'NOUN', 'PROPN', 'VERB', 'ADJ'} and not t.is_stop:
            kws.append(t.lemma_.lower())
    # Deduplicate and keep non-trivial tokens
    return sorted(list({k for k in kws if len(k) > 2}))

def _skill_matches(resume_text: str) -> int:
    text = resume_text.lower()
    count = 0
    for s in SKILL_LIST:
        if s in text:
            count += 1
    return count

def score_resume(jd_text: str, resume_text: str, vectorizer: TfidfVectorizer, nlp) -> Dict[str, float]:
    # TF-IDF cosine similarity
    tfidf_matrix = vectorizer.transform([jd_text, resume_text]).toarray()
    tfidf_score = _cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    # Keyword coverage
    jd_keywords = _extract_keywords(nlp, jd_text)
    resume_toks = set(_lemmatized_tokens(nlp, resume_text))
    if jd_keywords:
        hits = sum(1 for k in jd_keywords if k in resume_toks)
        keyword_cov = hits / len(jd_keywords)
    else:
        keyword_cov = 0.0

    # Skill bonus (normalize to 0–1 with simple cap)
    matches = _skill_matches(resume_text)
    # 8+ distinct skills is “strong”; cap at 1.0
    skill_bonus = min(matches / 8.0, 1.0)

    final_score = (
        WEIGHTS['tfidf'] * tfidf_score +
        WEIGHTS['keyword_cov'] * keyword_cov +
        WEIGHTS['skill_bonus'] * skill_bonus
    )

    return {
        'tfidf': tfidf_score,
        'keyword_cov': keyword_cov,
        'skill_bonus': skill_bonus,
        'final_score': final_score
    }
