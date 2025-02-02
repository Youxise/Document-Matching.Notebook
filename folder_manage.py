import os
import streamlit as st

# Function to get a list of files in a folder
def list_files(folder_path):
    try:
        # List files, not directories
        return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    except FileNotFoundError:
        st.sidebar.error("The folder path does not exist.")
        return []
    
def list_folders(path):
    try:
        # List folders
        items = os.listdir(path) 
        folders = [f for f in items if os.path.isdir(os.path.join(path, f))] # exclue les files
        return folders
    except FileNotFoundError:
        st.sidebar.error("The specified path does not exist.")
        return [], []