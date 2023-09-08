import pandas as pd
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
        
        # EDA
        if st.checkbox("Show EDA"):
            show_eda(df)

        task = st.selectbox("Select Task", ["Classification", "Regression"])
        
        # Model Training and Evaluation
        target_col = st.selectbox("Select Target Column", df.columns)
        
        if task == "Classification":
            setup_classification(df, target_col)
        else:
            setup_regression(df, target_col)
def encode_categorical_column(df, target_col):
    # Check if the column is of object (string) type or categorical type
    if df[target_col].dtype == 'object' or df[target_col].dtype.name == 'object':
        label_encoder = LabelEncoder()
        df[target_col] = label_encoder.fit_transform(df[target_col])
# Function to show basic EDA
def show_eda(df):
    st.write("Shape:", df.shape)
    st.write("Summary:", df.describe())
    
# Function to setup classification
def setup_classification(df, target_col):
    setup(data=df, target=target_col, session_id=123)  
    best_model = compare_models()
    if st.button("Train Models"):        
        st.write("models:", pull())
        st.write(f"Best Model: {best_model}")
        
    inx=pd.DataFrame(pull())
    xx=inx.reset_index()
    selected_model = st.selectbox("Select Model for Training", list(xx['index']))
    if st.button(f"Train {selected_model}"):
        model = create_model(selected_model)
        st.write(f"{selected_model} trained!")
        evaluate_model(model)        
        evaluation(best_model)
        

# Function to setup regression
def setup_regression(df, target_col):
    setup(data=df, target=target_col, session_id=123) 
    best_model = compare_models()
    if st.button("Train Models"):        
        st.write("models:", pull())
        st.write(f"Best Model: {best_model}")
    inx=pd.DataFrame(pull())
    xx=inx.reset_index()    
    selected_model = st.selectbox("Select Model for Training",list(xx['index']) )
    if st.button(f"Train {selected_model}"):
        model = create_model(selected_model)
        st.write(f"{selected_model} trained!")
        evaluate_model(model)    
        evaluation(best_model)
        
def evaluation(best_model):
    st.write("evaluation:", evaluate_model(best_model))        
if __name__ == "__main__":
    main()
    # Go to TERMINAL and run streamlit run pycaret.py