import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
from textblob import TextBlob

class EmailAnalyzer:
    def __init__(self):
        # Expanded trusted domains
        self.trusted_domains = [
            "google.com", "amazon.com", "apple.com", "microsoft.com",
            "paypal.com", "facebook.com", "twitter.com", "linkedin.com",
            "github.com", "netflix.com", "adobe.com"
        ]
        
        # Expanded suspicious keywords
        self.suspicious_keywords = [
            "urgent", "password", "verify now", "click here", "account suspended",
            "lottery", "winner", "claim now", "limited time", "act now",
            "bank transfer", "inheritance", "cryptocurrency", "investment opportunity"
        ]
        
        # Spam keywords
        self.spam_keywords = [
            "win", "free", "prize", "money", "urgent", "limited offer",
            "congratulations", "bitcoin", "investment", "lottery",
            "casino", "debt", "credit", "loan", "viagra", "pharmacy"
        ]

    def detect_suspicious_domain(self, sender):
        if "@" in sender:
            domain = sender.split("@")[1].lower().strip()
            return domain not in self.trusted_domains
        return True

    def detect_suspicious_keywords(self, text):
        text = str(text).lower()
        return any(keyword in text for keyword in self.suspicious_keywords)

    def categorize_email(self, subject):
        subject = str(subject).lower()
        if any(word in subject for word in ["sale", "discount", "off", "deal"]):
            return "Promotions"
        elif any(word in subject for word in ["invoice", "receipt", "order", "shipping"]):
            return "Transactions"
        else:
            return "Other"

    def calculate_risk_score(self, row):
        score = 0
        if row['suspicious_domain']:
            score += 50
        if row['suspicious_keywords']:
            score += 30
        if len(str(row['Subject'])) > 100:  # Long subjects can be suspicious
            score += 10
        return min(score, 100)  # Cap at 100

    def analyze_emails(self, df):
        # Basic cleaning and date conversion
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Core analysis
        df = self._analyze_phishing_risk(df)
        df = self._analyze_content(df)
        df = self._analyze_engagement(df)
        df = self._analyze_sentiment(df)
        df = self._analyze_links(df)
        
        return df

    def _analyze_phishing_risk(self, df):
        df['suspicious_domain'] = df['Sender'].apply(self.detect_suspicious_domain)
        df['suspicious_keywords'] = df['Subject'].apply(self.detect_suspicious_keywords)
        df['risk_score'] = df.apply(self.calculate_risk_score, axis=1)
        return df

    def _analyze_content(self, df):
        df['category'] = df['Subject'].apply(self.categorize_email)
        df['is_spam'] = df['Subject'].apply(self.detect_spam)
        df = self._cluster_emails(df)
        return df

    def _analyze_engagement(self, df):
        df['has_links'] = df['Message_body'].apply(lambda x: bool(re.findall(r'http[s]?://\S+', str(x))))
        df['is_reply'] = df['Subject'].str.contains('RE:', case=False, na=False)
        df['word_count'] = df['Message_body'].apply(lambda x: len(str(x).split()))
        return df

    def _analyze_sentiment(self, df):
        """Analyze sentiment of email subjects and bodies."""
        # Convert columns to string type and handle NaN values
        df['Subject'] = df['Subject'].fillna('').astype(str)
        df['Message_body'] = df['Message_body'].fillna('').astype(str)
        
        # Analyze sentiment
        df['subject_sentiment'] = df['Subject'].apply(self._get_sentiment)
        df['body_sentiment'] = df['Message_body'].apply(self._get_sentiment)
        return df

    def _analyze_links(self, df):
        df['links'] = df['Message_body'].apply(self._extract_links)
        df['suspicious_links'] = df['links'].apply(self._detect_suspicious_links)
        return df

    def _cluster_emails(self, df):
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['Subject'].fillna(''))
        kmeans = KMeans(n_clusters=5, random_state=42)
        df['cluster'] = kmeans.fit_predict(tfidf_matrix)
        return df

    def visualize_insights(self, df):
        self._plot_category_distribution(df)
        self._plot_volume_trends(df)
        self._plot_top_senders(df)
        self._plot_risk_scores(df)
        plt.show()

    def _plot_category_distribution(self, df):
        plt.figure(figsize=(10, 6))
        df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Email Category Distribution')
        plt.axis('equal')

    def _plot_volume_trends(self, df):
        plt.figure(figsize=(12, 6))
        daily_volume = df.groupby(df['Date'].dt.date).size()
        daily_volume.plot(kind='line', marker='o')
        plt.title('Daily Email Volume')
        plt.xlabel('Date')
        plt.ylabel('Number of Emails')
        plt.xticks(rotation=45)

    def _plot_top_senders(self, df):
        plt.figure(figsize=(12, 6))
        top_senders = df['Sender'].value_counts().head(10)
        sns.barplot(x=top_senders.values, y=top_senders.index)
        plt.title('Top 10 Email Senders')
        plt.xlabel('Number of Emails')

    def _plot_risk_scores(self, df):
        plt.figure(figsize=(10, 6))
        plt.scatter(df.index, df['risk_score'], alpha=0.5)
        plt.title('Email Risk Scores')
        plt.xlabel('Email Index')
        plt.ylabel('Risk Score')

    def generate_insights(self, df):
        return {
            'total_emails': len(df),
            'high_risk_emails': len(df[df['risk_score'] > 70]),
            'suspicious_domains': len(df[df['suspicious_domain']]),
            'category_distribution': df['category'].value_counts().to_dict(),
            'top_senders': df['Sender'].value_counts().head(5).to_dict(),
            'daily_volume': df.groupby(df['Date'].dt.date).size().to_dict()
        }

    def _get_sentiment(self, text):
        """Analyze sentiment of text, handling non-string inputs."""
        try:
            # Convert to string and handle NaN/None values
            if pd.isna(text):
                return 0
            text = str(text)
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        except:
            return 0

    def _extract_links(self, text):
        return re.findall(r'http[s]?://\S+', str(text))

    def _detect_suspicious_links(self, links):
        suspicious_domains = ["malware.com", "phishing.com", "malicious.com"]
        return any(domain in link for domain in suspicious_domains for link in links)

    def detect_spam(self, text):
        """Detect if a text is likely spam based on keywords and patterns."""
        if pd.isna(text):
            return False
            
        text = str(text).lower()
        
        # Check for spam keywords
        if any(keyword in text for keyword in self.spam_keywords):
            return True
            
        # Check for excessive punctuation
        if text.count('!') > 3 or text.count('$') > 2:
            return True
            
        # Check for ALL CAPS words
        words = text.split()
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        if caps_words > len(words) * 0.5:  # If more than 50% words are caps
            return True
            
        return False