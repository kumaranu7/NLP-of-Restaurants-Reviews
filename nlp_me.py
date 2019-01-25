#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing tsv dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
'''here delimiter serves as identification of tab and make it readable in csv 
format whereas quoting = 3 are used to identify double quotes'''
#cleaning the dataset-> stemming-> tokenization
import re
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])  
'''sub removes unwanted punctuations and here we keep'''
'''every letter a to z and A to Z, and replace unwanted by space'''
'''making every letter in lower case'''
review = review.lower()
'''removing non-significant words like the, that, in, or i.e. preposition
as in the end the sparse matrix every word has its own column'''
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords                                              
'''here 'review' is a string
so using split method it will get converted into list'''
review = review.split()
review = [word for word in review if not word in set(stopwords.words('english'))] 
'''first will go through different 
words by for loop, and if they are stopwords 
list then remove them; set() is added to increase the computation speed'''

                                    