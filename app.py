import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import re


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

def set_page_config():
    st.set_page_config(
        page_title="Bedok Reservoir Park Discourse Analysis",
        page_icon="💭",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .block-container {padding-top: 1rem;}
        h1 {color: #1E88E5;}
        h2 {color: #3949AB; font-size: 1.6rem;}
        h3 {color: #5E35B1;}
        .stTextArea label {font-size: 1.1rem;}
        </style>
    """, unsafe_allow_html=True)

def analyze_discourse(df):
    # Custom stopwords with categories
    custom_stops = {
        'location': ['bedok', 'reservoir', 'singapore'],
        'common': ['one', 'like', 'dont', 'really', 'around', 'thanks', 'place', 'think', 'was'],
        'time': ['today', 'yesterday', 'now', 'time', 'day', 'night', 'evening', 'morning']
    }
    
    all_stops = set(stopwords.words('english') + 
                   [word for category in custom_stops.values() for word in category])
    
    def clean_text(text):
        """Clean and tokenize text"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        # Lemmatize and filter tokens
        return [lemmatizer.lemmatize(t) for t in tokens if t not in all_stops and len(t) > 2]
    
    # Prepare data
    df['clean_tokens'] = df['text'].apply(clean_text)
    df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df['year_month'] = df['created_utc'].dt.strftime('%Y-%m')
    df['year'] = df['created_utc'].dt.year
    df['month'] = df['created_utc'].dt.month
    df['day_of_week'] = df['created_utc'].dt.day_name()
    
    return df

def show_keyword_analysis(df):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get all tokens and frequencies
        all_tokens = [token for tokens in df['clean_tokens'] for token in tokens]
        word_freq = Counter(all_tokens).most_common(10)
        word_df = pd.DataFrame(word_freq, columns=['word', 'count'])
        
        # Create enhanced bar chart
        fig = px.bar(
            word_df,
            x='word',
            y='count',
            title='Most Frequent Words in Discussions',
            color='count',
            color_continuous_scale='Viridis',
            labels={'count': 'Frequency', 'word': 'Word'}
        )
        fig.update_layout(
            title_x=0.5,
            showlegend=False,
            hoverlabel=dict(bgcolor="white"),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Word co-occurrence
        word_pairs = []
        for tokens in df['clean_tokens']:
            if len(tokens) >= 2:
                for i in range(len(tokens)-1):
                    word_pairs.append(f"{tokens[i]}_{tokens[i+1]}")
        
        top_pairs = Counter(word_pairs).most_common(5)
        st.subheader("Common Word Pairs")
        for pair, count in top_pairs:
            words = pair.split('_')
            st.write(f"• {words[0]} + {words[1]}: {count} times")

def show_sentiment_analysis(df):
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced sentiment distribution
        fig_dist = px.histogram(
            df,
            x='sentiment',
            nbins=50,
            color_discrete_sequence=['#3366cc'],
            title='Distribution of Sentiment Scores',
            labels={'sentiment': 'Sentiment Score', 'count': 'Number of Posts'}
        )
        fig_dist.update_layout(
            title_x=0.5,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_dist)
    
    with col2:
        # Sentiment by day of week
        dow_sentiment = df.groupby('day_of_week')['sentiment'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        fig_dow = px.bar(
            x=dow_sentiment.index,
            y=dow_sentiment.values,
            title='Average Sentiment by Day of Week',
            labels={'x': 'Day', 'y': 'Average Sentiment'},
            color=dow_sentiment.values,
            color_continuous_scale='RdBu'
        )
        fig_dow.update_layout(title_x=0.5, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dow)

    # Sentiment timeline
    monthly_sentiment = df.groupby('year_month').agg({
        'sentiment': ['mean', 'count', 'std']
    }).reset_index()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_sentiment['year_month'],
        y=monthly_sentiment['sentiment']['mean'],
        mode='lines+markers',
        name='Average Sentiment',
        line=dict(color='#3366cc'),
        error_y=dict(
            type='data',
            array=monthly_sentiment['sentiment']['std'],
            visible=True,
            color='#3366cc'
        )
    ))
    fig_trend.update_layout(
        title='Sentiment Trends Over Time',
        title_x=0.5,
        xaxis_title='Month',
        yaxis_title='Average Sentiment',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_trend)


def show_content_analysis(df):
    """Display filtered and categorized content with simpler interface"""
    st.subheader("Content Analysis")
    
    view_option = st.selectbox(
        "View posts by:",
        ["Most Discussed Topics", "Most Positive", "Most Negative", "Recent Posts"]
    )
    
    if view_option == "Most Positive":
        filtered_df = df.nlargest(5, 'sentiment')
        st.write("### Most Positive Posts")
        
    elif view_option == "Most Negative":
        filtered_df = df.nsmallest(5, 'sentiment')
        st.write("### Most Negative Posts")
        
    elif view_option == "Recent Posts":
        filtered_df = df.sort_values('created_utc', ascending=False).head(5)
        st.write("### Recent Posts")
        
    else:  
        filtered_df = df[df['text'].str.len() > 100].sort_values('created_utc', ascending=False).head(5)
        st.write("### Most Substantial Discussions")
    
    for _, post in filtered_df.iterrows():
        with st.expander(f"Post from {post['created_utc'].strftime('%Y-%m-%d')}"):
            # Show sentiment with emoji
            sentiment_emoji = "😊" if post['sentiment'] > 0 else "😐" if post['sentiment'] == 0 else "😟"
            st.markdown(f"**Sentiment:** {sentiment_emoji} ({post['sentiment']:.2f})")
            
            # Show the actual post text
            st.write(post['text'])
            
            # Show key words if any
            if len(post['clean_tokens']) > 0:
                st.markdown(f"**Key words:** {', '.join(post['clean_tokens'][:5])}")

def main():
    set_page_config()
    
    st.title("💭 Bedok Reservoir Reddit Discourse Analysis")
    
    # Load and process data
    try:
        df = pd.read_csv('BRP_reddit_data.csv')
        df = analyze_discourse(df)
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "🔑 Key Themes & Keywords",
        "😊 Sentiment Analysis",
        "🔍 Content Examples"
    ])
    
    with tab1:
        show_keyword_analysis(df)
    
    with tab2:
        show_sentiment_analysis(df)
    
    with tab3:
        show_content_analysis(df)



def main():
    set_page_config()
    
    st.title("💭 Bedok Reservoir Reddit Discourse Analysis")
    st.markdown("---")
    
    # Load and process data
    try:
        df = pd.read_csv('BRP_reddit_data.csv')
        df = analyze_discourse(df)
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        return
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Posts", len(df))
    with col2:
        st.metric("Average Sentiment", f"{df['sentiment'].mean():.2f}")
    with col3:
        st.metric("Time Span", f"{df['created_utc'].min().strftime('%Y')} - {df['created_utc'].max().strftime('%Y')}")
    with col4:
        st.metric("Unique Words", len(set([word for tokens in df['clean_tokens'] for word in tokens])))
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "🔑 Key Themes & Keywords",
        "😊 Sentiment Analysis",
        "🔍 Content Analysis"
    ])
    
    with tab1:
        show_keyword_analysis(df)
    
    with tab2:
        show_sentiment_analysis(df)
    
    with tab3:
        show_content_analysis(df)
  

if __name__ == "__main__":
    main()