import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_kernels
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

# Initial dataset by using a CSV file of ticket categories and descriptions
# In a full-scale platform these would be stored in the database
df_main = pd.read_csv("./sample-tickets-aidesk.csv", index_col=False)
df_main.set_index('ID', inplace=True)

# Creates a dataframe with the correct category
# For example, creating a dataframe only containing tickets under the Furniture category
df_correct_cat = df_main.loc[df_main['CATEGORY'] == 'Furniture']

# Output of the tickets that match the donor's category
# In a full-scale platform implementation, this filtering may happen in the core
# backend code instead of the recommendation system code
print(df_correct_cat)

# A list of the keywords entered by the donor on their profile
# In a full-scale platform these would be stored in the database
donor_keywords = ['Wood furniture', 'Bed', 'Bed Frame']
# Create a clean string of all the keywords on the donors profile
donor_keywords_str = ' '.join(donor_keywords).lower()

# A list to store the description bodies of the tickets to be used in the recommendation system
descriptions = []
# Data cleaning: All description bodies to lowercase and removing any stopwords, such as personal pronouns
for index, rows in df_correct_cat.iterrows():
  desc = rows['DESCRIPTION'].lower()
  desc = ' '.join([word for word in desc.split() if word not in stopwords])
  descriptions.append(desc)

# Creating a CountVectorizer based on the description bodies of the tickets to be used
# This simple code example uses characters, which will likely not be the case in a full-scale platform implementation
cv = CountVectorizer(analyzer='char')
cv.fit(descriptions)

# Uses cosine similarity between the string of the donor keywords and the description body of each ticket to calculate a match index between 0 and 1, where 1 is a perfect match
matches = pairwise_kernels(
  cv.transform([donor_keywords_str]),
  cv.transform(descriptions),
  metric='cosine'
)

# Adds a new column to the dataframe that indicates the match index of every ticket to the donor's keywords
df_correct_cat['Match'] = matches.ravel()

# Output matching tickets sorted by match index value instead of ID
# Sorted in descending order, so donors see higher matching tickets first on their list
print(df_correct_cat.sort_values('Match', ascending=False))