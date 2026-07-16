import pandas as pd

file = "Data/bias_article_details.csv"
df = pd.read_csv(file)
print(df['publish_date'].dtype)