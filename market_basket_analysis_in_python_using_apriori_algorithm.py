
'''

## Market Basket Analysis in Python using Apriori Algorithm
## Associate Rule Learning

Apriori is an algorithm for frequent item set mining and association rule learning over relational databases. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the database. The frequent item sets determined by Apriori can be used to determine association rules which highlight general trends in the database: this has applications in domains such as market basket analysis.

To make it simle it is used to find the association between two objects.

# Importing libraries
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""# Loading Dataset"""

df = pd.read_csv("Groceries_dataset[1].csv")

df.shape

df.describe()

df.info()

df.notnull().sum()

df.isna().sum()

"""## NO MISSING VALUES"""

df.head()

#setting index as Date
df.set_index('Date',inplace = True)

df.head()

#converting date into a particular format
df.index=pd.to_datetime(df.index)

df.head()

df.shape

#gathering information about products
total_item = len(df)
total_days = len(np.unique(df.index.date))
total_months = len(np.unique(df.index.year))
print(total_item,total_days,total_months)

"""### Total 38765 items sold in 728 days throughout 24 months"""

plt.figure(figsize=(15,5))
sns.barplot(x = df.itemDescription.value_counts().head(20).index, y = df.itemDescription.value_counts().head(20).values, palette = 'Spectral')
plt.xlabel('itemDescription', size = 15)
plt.xticks(rotation=45)
plt.ylabel('Count of Items', size = 15)
plt.title('Top 20 Items purchased by customers', color = 'green', size = 20)
plt.show()

df['itemDescription'].value_counts()

#grouping dataset to form a list of products bought by same customer on same date
df=df.groupby(['Member_number','Date'])['itemDescription'].apply(lambda x: list(x))

df.head(10)

#apriori takes list as an input, hence converting dtaset to a list
transactions = df.values.tolist()
transactions[:10]

#applying apriori
from apyori import apriori
rules = apriori(transactions, min_support=0.00030,min_confidence = 0.05,min_lift = 2,min_length = 2)
results = list(rules)
results

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
ordered_results = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

ordered_results

