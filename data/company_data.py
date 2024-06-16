import re
import pandas as pd
from transformers import pipeline

# Define scoring weights
weights = {
    'team_size': 0.2,
    'market_opportunity': 0.2,
    'innovation': 0.15,
    'business_model': 0.15,
    'scalability': 0.1,
    'traction': 0.2
}

# Define the scoring criteria
criteria = {
    'team_size': {
        "Single founder, no team, no experience": 1,
        "Team of 2+, little to no experience": 2,
        "Complementary team with some founders having significant work experience": 3,
        "Serial Entrepreneur(s) with no exits": 4,
        "Serial Entrepreneur(s) with multiple exits": 5
    },
    'market_opportunity': {
        "No market need, unclear problem that they are solving. Poor customer identification.": 1,
        "Product solves a problem with a midsize market, well served by competitors": 2,
        "Product solves a problem with a large market, well served by some competitors": 3,
        "Product solves a problem and has an attractive niche in a large market. Good value proposition to customers. Clear customer identification with unique positioning.": 4,
        "Product solves a problem and has an attractive niche in a large market. Very strong value proposition to customers. Clear customer identification with unique positioning in mostly untapped market (more than or equals to USD 1 billion).": 5
    },
    'innovation': {
        "No to Low innovation - localization of proven business models without change": 1,
        "Low innovation - localization of proven business model adapted to some markets": 2,
        "Some innovation - significant improvement of existing solution": 3,
        "Some unique IP, patents or data (pending patent)": 4,
        "Very strong innovation (IP / data)": 5
    },
    'business_model': {
        "Business model is impossible to realize": 1,
        "Hints at possible business model, financial projections need to be worked on": 2,
        "Business model explained, but not validated. Financial projections may or may not be available": 3,
        "Business model explained and first validation / tests are successful with real customers.": 4,
        "Good revenue model / business model is defined and has been validated with large number of customers": 5
    },
    'scalability': {
        "Solution is very manual / manpower heavy - no chance at scalability": 1,
        "Solution has the potential to scale beyond 1 city / small country, but has some issues to scale": 2,
        "Solution has no issues to scale globally or within home country but scaling has not started": 3,
        "Great potential to scale globally and has started to establish good networks in these countries but scaling has not started yet.": 4,
        "Has become well established in 1 City / small country. Has not started to scale anywhere else. No issue to scale otherwise": 5
    },
    'traction': {
        "Very little traction (social media engagement, landing page collecting potential customer info)": 1,
        "Prototype testing with initial customers (Beta testing)": 2,
        "Generating first revenues with paying customers. Has clear milestones and KPIs": 3,
        "Generating moderate revenue with paying customers. Has developed a lot of interest / relationships with key market participants or have significant traffic and engagement with customers.": 4,
        "Sustainable business (significant profit)": 5
    }
}

# Define sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# Function to calculate weighted score
def calculate_weighted_score(row):
    score = (
        row['team_size'] * weights['team_size'] +
        row['market_opportunity'] * weights['market_opportunity'] +
        row['innovation'] * weights['innovation'] +
        row['business_model'] * weights['business_model'] +
        row['scalability'] * weights['scalability'] +
        row['traction'] * weights['traction']
    )
    return score

# Function to extract information from the input text
def extract_information(text):
    def safe_extract(pattern, text):
        match = re.search(pattern, text)
        return match.group(1) if match else 'nan'

    team_size = safe_extract(r'Team Size: (.*)', text)
    market_opportunity = safe_extract(r'Market Opportunity / Problem to be solved: (.*)', text)
    innovation = safe_extract(r'Innovation: (.*)', text)
    business_model = safe_extract(r'Business Model: (.*)', text)
    scalability = safe_extract(r'Scalability: (.*)', text)
    traction = safe_extract(r'Traction: (.*)', text)

    return {
        'team_size': criteria['team_size'].get(team_size, 0),
        'market_opportunity': criteria['market_opportunity'].get(market_opportunity, 0),
        'innovation': criteria['innovation'].get(innovation, 0),
        'business_model': criteria['business_model'].get(business_model, 0),
        'scalability': criteria['scalability'].get(scalability, 0),
        'traction': criteria['traction'].get(traction, 0)
    }

# Function to determine eligibility
def determine_eligibility(text):
    info = extract_information(text)
    weighted_score = calculate_weighted_score(info)
    eligibility = 'ELIGIBLE' if weighted_score >= 2.9 else 'NOT ELIGIBLE'
    paragraph = '\n'.join([f"{k}: {v}" for k, v in info.items()])
    sentiment = sentiment_analysis(paragraph)[0]
    sentiment_label = sentiment['label']
    sentiment_score = sentiment['score']

    return {
        'text': text,
        'label': eligibility,
        'sentiment_label': sentiment_label,
        'sentiment_score': sentiment_score,
        'weighted_score': weighted_score
    }

# Example input text
input_text = """Company Name: Startup 1
Gender: FEMALE
Company Description: Dropbox lets you save and access all your files and photos in one place for easy sharing. Easily share files & access team content from your computer, mobile or any web browser.
Company Website: https://www.dropbox.com/
Job Titles: Chief Operating Officer (COO)/ Head of Operations
Business Model: nan
Revenue: $50,001 - $250,000 (USD)
Profit: Not generating profit yet
Total External Funding: 4000000
Notable Investors: Y Combinator, Sequioa Capital
Competition Region: North America

Team Size: Complementary team with some founders having significant work experience
Market Opportunity / Problem to be solved: Product solves a problem and has an attractive niche in a large market. Very strong value proposition to customers. Clear customer identification with unique positioning in mostly untapped market (more than or equals to USD 1 billion).
Innovation: Some unique IP, patents or data (pending patent)
Business Model: Good revenue model / business model is defined and has been validated with large number of customers
Scalability: Solution has no issues to scale globally or within home country but scaling has not started
Traction: Prototype testing with initial customers (Beta testing)
"""

# Determine eligibility and store the result
result = determine_eligibility(input_text)

# Create a DataFrame from the result
df = pd.DataFrame([result])

# Save the DataFrame to a CSV file
df.to_csv('eligibility_results.csv', index=False)

# Print the result
print(result)