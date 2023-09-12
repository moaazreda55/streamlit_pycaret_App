import pandas as pd
import matplotlib.pyplot as plt
from setuptools import setup
import pycaret 
import streamlit as st
from pycaret.classification import *
from pycaret.regression import *
from sklearn.preprocessing import LabelEncoder


# Main function to control app flow
def main():
    st.title("Streamlit + PyCaret App")
    
    # Load data
    data_upload = st.file_uploader("Upload Dataset (CSV only)", type=["csv"])
    if data_upload:
        df = pd.read_csv(data_upload)
        st.dataframe(df)
        columns_to_drop = st.multiselect("Select columns to drop:", df.columns)
        drop_coulmn(df,columns_to_drop)
        
        # EDA
        if st.checkbox("Show EDA"):
            show_eda(df)

        
        for column in df.columns:
            global technique
            if column not in columns_to_drop:
                st.write(f"#### Handling missing values in '{column}':")
                technique = st.selectbox(
                    f"What to do with missing values in '{column}'?",
                    ('most frequent', 'Missing') if df[column].dtype == 'object' else ('mean', 'median', 'mode')
                )

            handle_missing_values(df, column, technique)

        
        
        # Model Training and Evaluation
        target_col = st.selectbox("Select Target Column", df.columns)

        if df[target_col].dtype == 'object' :
           setup_classification(df, target_col) 
           
        else:
            setup_regression(df, target_col)
        
            
        st.write("dtypes:", df.dtypes)   
        st.write("null values:", df.isnull().sum())
        
def drop_coulmn(df,columns_to_drop):
     df.drop(columns=columns_to_drop, axis=1, inplace=True) 
    
def handle_missing_values(df, column, technique):
    if df[column].dtype == 'object':
        if technique == 'most frequent':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna('Missing', inplace=True)
    else:
        if technique == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif technique == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif technique == 'mode':
            df[column].fillna(df[column].mode()[0], inplace=True)  
        else :
            pass 
          
def encode_categorical_column(df, target_col):
    # Check if the column is of object (string) type or categorical type
    if df[target_col].dtype == 'object' or df[target_col].dtype.name == 'object':
        label_encoder = LabelEncoder()
        df[target_col] = label_encoder.fit_transform(df[target_col])
# Function to show basic EDA
def show_eda(df):
    st.write("Shape:", df.shape)
    st.write("dtypes:", df.dtypes)
    st.write("null values:", df.isnull().sum())
    st.write("Summary:", df.describe())
    
#Function to setup classification
def setup_classification(df, target_col):
    encode_categorical_column(df, target_col)
    setup(data=df, target=target_col, session_id=123)  
    best_model = compare_models()
    if st.button("Train Models"):        
        st.write("models:", pull())
        st.write(f"Best Model: {best_model}")
        
    inx=pd.DataFrame(pull())
    xx=inx.reset_index()
    selected_model = st.selectbox("Select Model for Training", list(xx['index']))
    if st.button(f"Train {selected_model}"):
        
        st.write(f"{selected_model} trained!")
                
        evaluation(best_model)
        

# Function to setup regression
def setup_regression(df, target_col):
    encode_categorical_column(df, target_col)
    setup(data=df, target=target_col, session_id=123) 
    best_model = compare_models()
    if st.button("Train Models"):        
        st.write("models:", pull())
        st.write(f"Best Model: {best_model}")
    inx=pd.DataFrame(pull())
    xx=inx.reset_index()    
    selected_model = st.selectbox("Select Model for Training",list(xx['index']) )
    if st.button(f"Train {selected_model}"):
       
        st.write(f"{selected_model} trained!")
            
        evaluation(best_model)
        
def evaluation(best_model):
      
    plot_model(best_model)
    st.pyplot(plt.gcf())
    plt.clf()
          
if __name__ == "__main__":
    main()
    # Go to TERMINAL and run streamlit run pycaret.py
