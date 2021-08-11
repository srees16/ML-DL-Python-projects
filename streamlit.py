#streamlit project

'''
practice streamlit:
    https://www.youtube.com/watch?v=B0MUXtmSpiA
    https://www.youtube.com/watch?v=8M20LyCZDOY
    https://www.youtube.com/watch?v=Eai1jaZrRDs
    https://www.youtube.com/watch?v=xiBXspqs0dk
    https://www.youtube.com/watch?v=z5HfbXORZsg
    https://www.youtube.com/watch?v=iZUH1qlgnys
    https://www.youtube.com/watch?v=zYSDlbr-8V8
    
Ref: https://www.youtube.com/playlist?list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE, https://www.youtube.com/playlist?list=PLM8lYG2MzHmTATqBUZCQW9w816ndU0xHc, 	https://www.youtube.com/playlist?list=PLuU3eVwK0I9PT48ZBYAHdKPFazhXg76h5

container create sections horizontally, columns create sections vertically (referred as beta_container and beta_columns)
streamlit run pyfile.py to run streamlit app

plan elements/contents for the front-end:
- project title
- explanation
- dataset intro
- plots

annotations:
    st.cache

functions:
    st.title()
    st.text()
    st.header()
    st.beta_container()
    st.beta_columns()
    st.dataframe()
    st.table() # displays all samples
    st.json()
    st.text_input()
    st.text_area()
    st.date_input()
    st.time_input()
    st.checkbox()
    st.radio('colors', ['r', 'b', 'g'], index = 0)
    st.select_box()
    st.multiselect('colors', ['r', 'b', 'g'])
    st.slider()
    st.number_input('choose age', min_value = 10, max_value = 100, step = 1)
    st.line_chart()
    st.file_uploader()
    st.multiselect()
    st.image() # use Image.open('image_name.jpg') imported from PIL
    st.audio()
    st.button('Submit')
    st.error('an error occured')
    st.success('celebrate success')
    st.info('Info')
    st.exception(RuntimeError)
    st.warning('warning to you')

sidebar functions: works for widgets except write echo n spinner
    st.sidebar.header
    st.sidebar.slider
    st.sidebar.text_input
    st.sidebar.selectbox

plots:
    st.graphviz_chart() # for flow chart
'''

import pandas as pd
import numpy as np
import string
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

#@st.cache
def get_data(filename = None):
    x, y = make_classification(n_samples = 1000, n_features = 5, shuffle = True, random_state = 8)
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    x.columns = [list(string.ascii_uppercase)[i] for i in range(len(x.columns))]
    return x, y

x, y = get_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 7)

### streamlit
header = st.beta_container()
df = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

st.header('charts & plots')
st.line_chart(x[:20])
st.area_chart(x[['A', 'E']][20:40])
st.bar_chart(x[50:70])

st.sidebar.header('User input values')

# to write something in above containers, use with
with header:
    st.title('Welcome to Manhattan Project')
    st.text('analysis of project results')

# for dataset
with df:
    st.header('Here is the dataset')
    st.text('an open source dataset considered apt for the project')
    st.dataframe(data = x) # either of them works
    #st.table(x)
    st.subheader('distribution of values')
    st.bar_chart(x['A'].value_counts().head(50))

# for features
with features:
    st.header('My features')
    st.text('Columns/features that are considered for model training')
    st.markdown('''# here is h1 tag\n## here is h2 tag\n### here is h3 tag\n:sunglasses:\n ** bold text ** _ italics text _ \n *** bold & italics text ***\n''')
    st.beta_columns(len(x.columns)) # pass no of columns needed

# for model training
with model_training:
    st.header('Here is the model training results')
    st.text('Below results for display only')
    
    sel_col, disp_col = st.beta_columns(2)
    max_depth = sel_col.slider(label = 'Max depth of the tree?', min_value = 10, max_value =  200, value = 20, step = 1)
    estimators = sel_col.selectbox(label = 'How many trees', options = [5, 10, 15, 20, 30, 40, 50, 75, 100], index = 0)
    kernel = sel_col.selectbox(label = 'choose the split criteria', options = ['gini', 'entropy'])
    
    sel_col.text('Few input features are: ')
    sel_col.write(x.columns) # write can be used for multiple ways depending input object
    input_features = sel_col.text_input(label = 'Preferred input features', max_chars = 100) # , value = None # value is default value
    
    classifier = RandomForestClassifier(criterion = kernel, max_depth = max_depth, n_estimators = estimators)

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    
    disp_col.subheader('MAE is ')
    disp_col.write(mean_absolute_error(y_test, y_pred))
    
    disp_col.subheader('MSE is ')
    disp_col.write(mean_squared_error(y_test, y_pred))
    
    disp_col.subheader('R-squared score is ')
    disp_col.write(r2_score(y_test, y_pred))

st.title('Registration Form')
first, last = st.beta_columns(2)
first.text_input(label = 'First name')
last.text_input(label = 'Last name')
email, mobile = st.beta_columns([3, 1]) # list of values divide columns for both fields in 3:1 ratio
email.text_input(label = 'Email')
mobile.text_input(label = 'Mobile')
user, pw1, pw2 = st.beta_columns(3)
user.text_input(label = 'user name')
pw1.text_input(label = 'password')
pw2.text_input(label = 'retype password')
ch, _, sub = st.beta_columns(3)
ch.checkbox(label = 'I agree', value = False)
sub.button('Submit')

st.video(data = '/Users/sreekanths/.Trash/NtPM24RGRMT_IqW2 10.41.50 PM.mp4') # can add youtube vids too
st.map()
