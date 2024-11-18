import pandas as pd
import json
import numpy as np
from email_analyzer import EmailAnalyzer

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8,
                       np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.datetime64, pd.Timestamp)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def generate_detailed_insights(df):
    insights = {
        'basic_stats': {
            'total_emails': int(len(df)),
            'unique_senders': int(df['Sender'].nunique()),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}"
        },
        'risk_analysis': {
            'high_risk_emails': int(len(df[df['risk_score'] > 70])),
            'suspicious_domains': int(len(df[df['suspicious_domain']])),
            'average_risk_score': float(df['risk_score'].mean())
        },
        'categories': {k: int(v) for k, v in df['category'].value_counts().to_dict().items()},
        'engagement': {
            'emails_with_links': int(df['has_links'].sum()),
            'reply_rate': float((df['is_reply'].sum() / len(df)) * 100),
            'average_word_count': float(df['word_count'].mean())
        },
        'sentiment': {
            'positive_emails': int(len(df[df['subject_sentiment'] > 0])),
            'negative_emails': int(len(df[df['subject_sentiment'] < 0])),
            'neutral_emails': int(len(df[df['subject_sentiment'] == 0]))
        }
    }
    
    # Convert all values to serializable types
    return {k: convert_to_serializable(v) if isinstance(v, dict) 
            else convert_to_serializable(v) 
            for k, v in insights.items()}

def save_insights(insights):
    with open('email_insights.json', 'w') as f:
        json.dump(insights, f, indent=4)

def print_summary(insights):
    print("\n=== Email Analysis Summary ===")
    print(f"Total Emails: {insights['basic_stats']['total_emails']}")
    print(f"Date Range: {insights['basic_stats']['date_range']}")
    print(f"\nRisk Analysis:")
    print(f"- High Risk Emails: {insights['risk_analysis']['high_risk_emails']}")
    print(f"- Average Risk Score: {insights['risk_analysis']['average_risk_score']:.2f}")
    print("\nCategory Distribution:")
    for category, count in insights['categories'].items():
        print(f"- {category}: {count}")

def main():
    # Load data
    df = pd.read_csv('emails.csv')
    
    # Initialize analyzer
    analyzer = EmailAnalyzer()
    
    # Run analysis
    processed_df = analyzer.analyze_emails(df)
    
    # Generate insights
    insights = generate_detailed_insights(processed_df)
    
    # Save results
    processed_df.to_csv('emails_with_analytics.csv', index=False)
    save_insights(insights)
    
    # Generate visualizations
    analyzer.visualize_insights(processed_df)
    
    # Print summary
    print_summary(insights)

if __name__ == "__main__":
    main()