import praw
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os


load_dotenv()

def scrape_reddit_content():
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )

    urls = [
        "https://www.reddit.com/r/singaporehappenings/s/fF0YqdPjvC",
        "https://www.reddit.com/r/singapore/s/0cqMcwZvM4",
        "https://www.reddit.com/r/singapore/s/HbbU1svIp9/",
        "https://www.reddit.com/r/singapore/s/veAlf0CD5K",
        "https://www.reddit.com/r/singapore/s/qbEXtopGXl",
        "https://www.reddit.com/r/Fishing/s/gY3822crTB"
    ]

    data = []
    
    for url in urls:
        post = reddit.submission(url=url)
        data.append({
                'id': post.id,
                'type': 'post',
                'title': post.title,
                'text': post.selftext,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'score': post.score,
                'url': post.url,
                'author': getattr(post.author, 'name', '[deleted]'),
                'num_comments': post.num_comments,
                'post_id': post.id, 
                'parent_id': None    
            })

        if post.num_comments > 0:
            try:
                post.comments.replace_more(limit=None)
                for comment in post.comments.list():
                    if hasattr(comment, 'body'):  
                        data.append({
                            'id': comment.id,
                            'type': 'comment',
                            'title': post.title,
                            'text': comment.body,
                            'created_utc': datetime.fromtimestamp(comment.created_utc),
                            'score': comment.score,
                            'url': None,  
                            'author': getattr(comment.author, 'name', '[deleted]'),
                            'num_comments': 0,  
                            'post_id': post.id,
                            'parent_id': comment.parent_id
                        })
            except Exception as e:
                print(f"Error processing comments for post {post.id}: {str(e)}")

        df = pd.DataFrame(data)
        df = df.sort_values('created_utc')
        df = df.drop_duplicates(subset='id')
        
    return df


if __name__ == "__main__":
    try:
        print("Starting data collection...")
        reddit_df = scrape_reddit_content()

        reddit_df.to_csv('BRP_reddit_data.csv', index=False)
        
        print(f"\nData collection complete!")
        print(f"Total entries: {len(reddit_df)}")
        print(f"Posts: {len(reddit_df[reddit_df['type'] == 'post'])}")
        print(f"Comments: {len(reddit_df[reddit_df['type'] == 'comment'])}")
        print(f"Date range: {reddit_df['created_utc'].min()} to {reddit_df['created_utc'].max()}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")