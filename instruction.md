1. Phishing Risk Detection

Detect emails with suspicious domains and keywords, and assign a phishing risk score.

import pandas as pd

# Load dataset

file_path = "emails.csv" # Replace with your file path
emails_df = pd.read_csv(file_path)

# Trusted domains for detection

trusted_domains = ["google.com", "paypal.com", "amazon.com"]

def detect_suspicious_domain(sender):
if "@" in sender:
domain = sender.split("@")[-1].strip()
return domain not in trusted_domains
return True

suspicious_keywords = ["urgent", "password", "click here", "verify now"]

def detect_suspicious_keywords(content):
for keyword in suspicious_keywords:
if keyword.lower() in content.lower():
return True
return False

# Apply phishing risk detection

emails_df['suspicious_domain'] = emails_df['Sender'].apply(detect_suspicious_domain)
emails_df['suspicious_keywords'] = emails_df['Subject'].apply(detect_suspicious_keywords)

def calculate_phishing_risk(row):
score = 0
if row['suspicious_domain']:
score += 50
if row['suspicious_keywords']:
score += 50
return score

emails_df['phishing_risk_score'] = emails_df.apply(calculate_phishing_risk, axis=1)
print(emails_df[['Sender', 'Subject', 'phishing_risk_score']])

2. Email Volume Analytics

Analyze daily and weekly email trends.

# Convert Date column to datetime

emails_df['Date'] = pd.to_datetime(emails_df['Date'])

# Calculate daily email volume

daily_volume = emails_df.groupby(emails_df['Date'].dt.date).size()
print("Daily Email Volume:")
print(daily_volume)

# Calculate weekly email volume

weekly_volume = emails_df.groupby(emails_df['Date'].dt.isocalendar().week).size()
print("Weekly Email Volume:")
print(weekly_volume)

3. Sender Analytics

Identify top senders and flag unknown ones.

# Top 10 most frequent senders

top_senders = emails_df['Sender'].value_counts().head(10)
print("Top 10 Senders:")
print(top_senders)

# Flag unknown senders

known_contacts = ["trusted@domain.com", "friend@example.com"]

def is_unknown_sender(sender):
return sender not in known_contacts

emails_df['unknown_sender'] = emails_df['Sender'].apply(is_unknown_sender)
unknown_senders = emails_df[emails_df['unknown_sender']]
print("Emails from Unknown Senders:")
print(unknown_senders[['Sender', 'Subject']])

4. Content Categorization

Categorize emails into predefined categories based on their subjects.

def categorize_email(subject):
if "sale" in subject.lower() or "discount" in subject.lower():
return "Promotions"
elif "invoice" in subject.lower() or "receipt" in subject.lower():
return "Transactions"
elif "friend request" in subject.lower() or "follow" in subject.lower():
return "Social"
else:
return "Personal"

emails_df['category'] = emails_df['Subject'].apply(categorize_email)
print("Email Categories:")
print(emails_df['category'].value_counts())

5. Keyword and Sentiment Analysis

Extract keywords and analyze sentiment in email subjects.

from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

# Extract keywords

vectorizer = CountVectorizer(stop_words='english', max_features=10)
subject_keywords = vectorizer.fit_transform(emails_df['Subject'].dropna())
keywords = vectorizer.get_feature_names_out()
print("Top Keywords in Email Subjects:")
print(keywords)

# Sentiment analysis

def analyze_sentiment(content):
analysis = TextBlob(content)
return analysis.sentiment.polarity # Polarity ranges from -1 (negative) to 1 (positive)

emails_df['sentiment_score'] = emails_df['Subject'].apply(lambda x: analyze_sentiment(x) if pd.notnull(x) else 0)
print("Emails with Sentiment Scores:")
print(emails_df[['Subject', 'sentiment_score']])

6. Time-Based Analysis

Analyze peak times for receiving emails and suggest optimal times for managing them.

# Extract hour from the Date

emails_df['hour'] = emails_df['Date'].dt.hour

# Calculate peak hours

peak_hours = emails_df['hour'].value_counts().head(3)
print("Peak Email Hours:")
print(peak_hours)

# Visualize peak email times (optional)

import matplotlib.pyplot as plt

emails_df['hour'].value_counts().sort_index().plot(kind='bar', figsize=(10, 6))
plt.title("Email Volume by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Emails")
plt.xticks(rotation=0)
plt.show()

7. Unread Emails Analysis

Track the number and percentage of unread emails.

# Unread emails (example: Message_body is null)

emails_df['unread'] = emails_df['Message_body'].isnull()

# Count unread emails

total_unread = emails_df['unread'].sum()
total_emails = len(emails_df)
percentage_unread = (total_unread / total_emails) \* 100

print(f"Unread Emails: {total_unread} ({percentage_unread:.2f}%)")

# Count unread emails by category

unread_by_category = emails_df.groupby('category')['unread'].sum()
print("Unread Emails by Category:")
print(unread_by_category)

8. Email Response Time Analysis

Track response times to emails to identify delays.

# Assuming the dataset includes a 'Response_Date' column

emails_df['Response_Date'] = pd.to_datetime(emails_df['Response_Date'], errors='coerce')

# Calculate response time in hours

emails_df['response_time_hours'] = (emails_df['Response_Date'] - emails_df['Date']).dt.total_seconds() / 3600

# Average response time

average_response_time = emails_df['response_time_hours'].mean()
print(f"Average Response Time: {average_response_time:.2f} hours")

# Response time distribution

emails_df['response_time_hours'].dropna().plot(kind='hist', bins=20, figsize=(10, 6))
plt.title("Response Time Distribution")
plt.xlabel("Response Time (hours)")
plt.ylabel("Frequency")
plt.show()

9. Email Engagement Metrics

Analyze how often users interact with emails (e.g., read, replied, clicked links).

# Example engagement flags (update based on actual columns)

emails_df['clicked_link'] = emails_df['Message_body'].str.contains("http") # Assuming links in the body
emails_df['replied'] = emails_df['Subject'].str.contains("RE:") # Example reply detection

# Engagement metrics

clicked_links_count = emails_df['clicked_link'].sum()
replied_count = emails_df['replied'].sum()

print(f"Emails with Clicked Links: {clicked_links_count}")
print(f"Replied Emails: {replied_count}")

10. Spam Detection

Detect and analyze spam emails based on certain characteristics.

# Define spam keywords and patterns

spam_keywords = ["win", "free", "prize", "money", "urgent"]

def detect_spam(content):
for keyword in spam_keywords:
if pd.notnull(content) and keyword.lower() in content.lower():
return True
return False

# Apply spam detection

emails_df['is_spam'] = emails_df['Subject'].apply(detect_spam)

# Count spam emails

spam_count = emails_df['is_spam'].sum()
print(f"Number of Spam Emails: {spam_count}")

# Spam percentage

spam_percentage = (spam_count / len(emails_df)) \* 100
print(f"Percentage of Spam Emails: {spam_percentage:.2f}%")

11. Link Safety Analysis

Analyze links in emails for potential phishing or malware risks.

import re

# Extract links from email bodies

def extract_links(message_body):
if pd.notnull(message_body):
return re.findall(r'http[s]?://\S+', message_body)
return []

emails_df['links'] = emails_df['Message_body'].apply(extract_links)

# Mark emails with suspicious links (example: short URLs, known bad domains)

def detect_suspicious_links(links):
suspicious_domains = ["bit.ly", "tinyurl.com", "malicious.com"]
for link in links:
for domain in suspicious_domains:
if domain in link:
return True
return False

emails_df['suspicious_links'] = emails_df['links'].apply(detect_suspicious_links)
print("Emails with Suspicious Links:")
print(emails_df[emails_df['suspicious_links']])

12. Email Clustering

Use clustering algorithms to group similar emails.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Apply TF-IDF to email subjects

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(emails_df['Subject'].dropna())

# Perform K-Means clustering

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
emails_df['cluster'] = kmeans.fit_predict(tfidf_matrix)

print("Cluster Assignments:")
print(emails_df[['Subject', 'cluster']])

13. Sentiment Analysis

Analyze the sentiment of email subjects or bodies.

from textblob import TextBlob

def analyze_sentiment(text):
if pd.notnull(text):
return TextBlob(text).sentiment.polarity # Polarity: -1 (negative) to +1 (positive)
return 0

emails_df['subject_sentiment'] = emails_df['Subject'].apply(analyze_sentiment)
emails_df['body_sentiment'] = emails_df['Message_body'].apply(analyze_sentiment)

# Average sentiment

average_sentiment = emails_df['subject_sentiment'].mean()
print(f"Average Sentiment of Subjects: {average_sentiment:.2f}")

14. Email Summary Extraction

Summarize long emails to provide a quick overview.

from gensim.summarization import summarize

def extract_summary(message_body):
if pd.notnull(message_body) and len(message_body.split()) > 50: # Summarize only long texts
try:
return summarize(message_body, word_count=50)
except ValueError: # Handle short texts that can't be summarized
return message_body
return message_body

emails_df['summary'] = emails_df['Message_body'].apply(extract_summary)
print("Email Summaries:")
print(emails_df[['Message_body', 'summary']])
