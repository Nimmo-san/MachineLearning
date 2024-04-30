
import pandas as pd


#print(pd.DataFrame({'Yes' : [50, 21], 'No': [131, 2]}))
# Result
#    Yes   No
# 0   50  131
# 1   21    2


# Data Frame entries are not limited to integers

#print(pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']}))
#              Bob           Sue
# 0    I liked it.  Pretty good.
# 1  It was awful.        Bland.

# Creating a data frame with our own index
#print(pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
#              'Sue': ['Pretty good.', 'Bland.']},
#             index=['Product A', 'Product B']))
#                      Bob           Sue
# Product A    I liked it.  Pretty good.
# Product B  It was awful.        Bland

# Reading data files

filepath = './Machine Learning/melb_data.csv'
unfil_data = pd.read_csv(filepath, index_col=0)

# Prints the size of our data
print(unfil_data.shape)

# Contents of our data, the first five rows
print(unfil_data.head())
