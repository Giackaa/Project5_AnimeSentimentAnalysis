import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_page(url, ep, num_of_pages):
    all_comments = []
    for page in range (num_of_pages):
        # Scrape all pages (page param is offset, e.g. 0, 50, 100, 150)
        offset = page * 50
        print(f"Scraping page with offset {offset}...")
        url = url + f"&show={offset}"
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        comments = []

        # Each post is in a .forum-post element
        posts = soup.find_all("div", {"class": "forum-topic-message"})
        for post in posts:
            username = post.find("a", {"class": "ga-click"})
            comment_block = post.find("div", {"class": "content"})
            if comment_block:
            # Get quoted reply if it exists
                quote = comment_block.find("div", {"class": "quotetext"})
                quoted_user = None
                quoted_text = None
                if quote:
                    quoted_user = quote.get("data-user")
                    quoted_text = quote.get_text(strip=True)

                # Remove the quote from the HTML so it doesn't get mixed with the main comment
                    quote.decompose()

                reply = comment_block.find("div", {"class": "replied show"})
                if reply:
                    reply.decompose()

                # Get remaining comment (after quote is removed)
                main_comment = comment_block.text
            join_date = post.find("div", {"class": "userinfo joined"})
            post_num = post.find("div", {"class": "userinfo posts"})

            comments.append({
                "episode": f"Ep{ep}",
                "username": username.text if username else None,
                "comment": main_comment,
                "quoted_user": quoted_user,
                "quoted_text" : quoted_text,
                "join_date": join_date.text if join_date else None,
                "post_num": post_num.text if post_num else None
            })
        all_comments.extend(comments)

    return all_comments
file = r"C:\Users\jackl\OneDrive\Desktop\Project 5 - Anime Sentiments\mal_comments2.csv"

def clean_data (df):
    # Calculate the number of non-null values for each row
    df['non_null_count'] = df.apply(lambda x: x.count(), axis=1)

    # Sort by 'comment' and 'non_null_count' to keep the row with more data
    df_sorted = df.sort_values(by=['comment', 'non_null_count'], ascending=[True, False])

    # Drop duplicates based on the 'comment' column, keeping the first occurrence (which will be the one with the most non-null values due to sorting)
    df_cleaned = df_sorted.drop_duplicates(subset=['comment'], keep='first')

    # Drop the temporary 'non_null_count' column
    df_cleaned = df_cleaned.drop(columns=['non_null_count'])

    return df_cleaned

df = pd.read_csv(file)
df = clean_data(df)
df.to_csv(file, index=False)

