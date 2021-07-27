# Word Cloud

import pandas as pd
from PIL import Image
from os import path
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Reads 'Youtube04-Eminem.csv' file  
df = pd.read_csv(r'C:\Users\Srees\Desktop\WhatsApp Chat with Sh Sukeshini.txt', sep=r'[ap]m -', names=['time', 'message'])
  
# if your mobile phone has a 24 hour time format, then use the below code to import the file:
df1 = pd.read_csv('whatsapp.txt', sep=r'[0-9] -', names=['time', 'message'])

# Split the message further into: name and original message:
df2 = df['message'].str.split(":", expand=True,n=1)
df_all = pd.concat([df, df2], axis=1)
df_all = df_all.rename(columns={'message': 'total', 0:'name', 1:'message'})
df_all.drop('total', axis=1, inplace=True)

# Saving the messages which are in the time column instead of message column
df_all.loc[df_all.time.str.contains(r'[a-zA-Z]')==True, 'message'] = df_all[df_all.time.str.contains(r'[a-zA-Z]')==True].time
df_all.fillna(' ', inplace=True)

# Delete rows where name includes an activity on group #
df_all = df_all[df_all.name.str.contains("added|changed|created|left")==False]

# Store the text in a variable
text = ' '.join(df_all['message'])

# Remove stopwords if any (you can add more to this list)
STOPWORDS.update(['media', 'omitted', 'Srees', 'SreeS', 'Sukeshini'])

# Creating the word cloud
wc = WordCloud(background_color="white", max_words=2000, stopwords=STOPWORDS.add("said"))
wc.generate(text)
plt.imshow(wc)
wc.to_file("word_cloud.png")

###############################################################################################

