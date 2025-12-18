# app.py
import os
import uuid
import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename

from resume_parser import extract_text_from_pdf
from scoring import load_nlp, build_vectorizer, score_resume
from utils import ensure_dirs, clean_text

UPLOAD_FOLDER = 'data/resumes'
REPORT_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.secret_key = 'dev-secret-key'  # TODO: set env var in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

nlp = load_nlp()  # spaCy model
vectorizer = None
jd_text_cached = None

ensure_dirs([UPLOAD_FOLDER, REPORT_FOLDER])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    global vectorizer, jd_text_cached
    if request.method == 'POST':
        jd_text = request.form.get('job_description', '').strip()
        jd_file = request.files.get('jd_file')

        if jd_file and jd_file.filename and jd_file.filename.lower().endswith('.txt'):
            jd_text = jd_file.read().decode('utf-8', errors='ignore')

        if not jd_text:
            flash('Please paste or upload a job description (.txt).')
            return redirect(url_for('index'))

        files = request.files.getlist('resumes')
        resume_paths = []

        for f in files:
            if f and allowed_file(f.filename):
                fname = secure_filename(f.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                f.save(save_path)
                resume_paths.append(save_path)

        if not resume_paths:
            flash('Please upload at least one PDF resume.')
            return redirect(url_for('index'))

        jd_text_cached = clean_text(jd_text)
        # Build vectorizer on JD + resumes for balanced vocabulary
        corpus_texts = [jd_text_cached]
        resumes_text = []
        for path in resume_paths:
            text = extract_text_from_pdf(path)
            resumes_text.append(clean_text(text))
            corpus_texts.append(clean_text(text))

        vectorizer = build_vectorizer(corpus_texts)

        rows = []
        for path, rtext in zip(resume_paths, resumes_text):
            try:
                sc = score_resume(jd_text_cached, rtext, vectorizer, nlp)
            except Exception as e:
                sc = {'final_score': 0.0, 'tfidf': 0.0, 'keyword_cov': 0.0, 'skill_bonus': 0.0, 'error': str(e)}
            rows.append({
                'candidate': os.path.basename(path),
                'tfidf': round(sc['tfidf'], 4),
                'keyword_cov': round(sc['keyword_cov'], 4),
                'skill_bonus': round(sc['skill_bonus'], 4),
                'final_score': round(sc['final_score'], 4)
            })

        df = pd.DataFrame(rows).sort_values(by='final_score', ascending=False)

        # Save report
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f'hr_report_{ts}_{uuid.uuid4().hex[:6]}.csv'
        report_path = os.path.join(REPORT_FOLDER, report_name)
        df.to_csv(report_path, index=False)

        return render_template('results.html',
                               jd_text=jd_text_cached[:2000],
                               table=df.to_dict(orient='records'),
                               report_name=report_name)
    return render_template('index.html')
    
@app.route('/download/<report_name>')
def download_report(report_name):
    path = os.path.join(REPORT_FOLDER, report_name)
    if not os.path.exists(path):
        flash('Report not found.')
        return redirect(url_for('index'))
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
