import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

# load YouTube API key and required libraries
# ...

# get video transcript
def get_transcript(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.captions().list(
        part='id',
        videoId=video_id
    )
    response = request.execute()
    caption_id = response['items'][0]['id']
    
    request = youtube.captions().download(
        id=caption_id,
        tfmt='srt'
    )
    caption = request.execute()
    transcript = ""
    for line in caption.split('\n'):
        if line.isdigit() == False and line != '':
            transcript += line + ' '
    return transcript

# preprocess sentence
def preprocess(sent):
    sent_lower = sent.lower()
    sent_no_punc = ''.join(e for e in sent_lower if e.isalnum() or e.isspace())
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(sent_no_punc)
    filtered_tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemm_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    sent_preprocessed = ' '.join(lemm_tokens)
    return sent_preprocessed

# summarize transcript
def summarize(transcript):
    sentences = sent_tokenize(transcript)
    sentence_scores = {}
    for sent in sentences:
        sent_len = len(sent.split())
        if sent_len > 3 and sent_len < 50:
            sent_pre = preprocess(sent)
            for word in sent_pre.split():
                if word not in sentence_scores.keys():
                    sentence_scores[word] = 1
                else:
                    sentence_scores[word] += 1
    for word in sentence_scores.keys():
        sentence_scores[word] = sentence_scores[word] / len(sentence_scores.keys())
    summary_sentences = []
    for sent in sentences:
        sent_pre = preprocess(sent)
        sent_score = 0
        for word in sent_pre.split():
            if word in sentence_scores.keys():
                sent_score += sentence_scores[word]
        summary_sentences.append((sent, sent_score))
    summary_sentences = sorted(summary_sentences, key=lambda x: x[1], reverse=True)
    summary = ""
    for sent, score in summary_sentences[:5]:
        summary += sent + " "
    return summary
