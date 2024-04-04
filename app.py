# Importing the required packages

import numpy as np
import streamlit as st
import os
import hashlib
from shutil import move
from scipy.spatial import distance
import tkinter as tk
from tkinter import filedialog

from sentence_transformers import SentenceTransformer
import PyPDF2
import docx2txt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import pickle

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
img_model = VGG16(weights='imagenet', include_top=False)
img_ext = (".jpg", ".jpeg", ".tiff", ".png")

# Set document text similarity score
cosine_score = 0.9

# Set Image similarity score
img_similarity_score = 0.95

document_folder = ""
duplicates_folder = ""
st.set_page_config(layout="wide", page_title="Duplicate files remover", page_icon=":newspaper:")


# Defining the SHA Algorithm


def create_hash_database(folder_path):
    hash_database = {}  # creating a dictionary for hash_database
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'rb') as f:
                    md5_hash = hashlib.md5(f.read()).hexdigest()
                    sha256_hash = hashlib.sha256(f.read()).hexdigest()
                hash_database[filename] = (md5_hash, sha256_hash)

            except FileNotFoundError:
                st.info(f"Error: File '{filename}' not found. Hence skipping...")
    return hash_database


# Create a folder if  folder doesn't exist
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Check for duplicate in the file
def check_duplicate(filename, hash_database):
    if filename in hash_database:
        return True

    return False


def move_duplicate_llm(filename, duplicate_file, similarity_score, source_folder, destination_folder):
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)
    if os.path.exists(source_path):
        try:
            move(source_path, destination_path)
            st.info(
                f"Duplicate '{filename}' moved to duplicate folder with similarity score of {similarity_score} with file {duplicate_file}")

        except Exception as e:
            st.info(f"Error moving '{filename}': {e}")


def move_duplicate(filename, source_folder, destination_folder, duplicate_filename):
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)
    if os.path.exists(source_path):
        try:
            move(source_path, destination_path)
            st.info(f"Duplicate '{filename}' moved to duplicate folder as it is duplicate to {duplicate_filename}")

        except Exception as e:
            st.info(f"Error moving '{filename}': {e}")


def extract_text(path):
    text = ""
    if path.endswith('.docx'):
        text = docx2txt.process(path)

    elif path.endswith('.pdf'):
        with open(path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    else:
        pass

    return text


def cosine_similarity_img(img_1_embd, img_2_embd):
    cosine_similarity = 1 - distance.cosine(img_1_embd, img_2_embd)
    return cosine_similarity


# Generate Image Embedding
def img_embedding(path):
    # Preprocess image
    img = image.load_img(path, target_size=(224, 224))  # Resize to match VGG16 input size
    img_array = image.img_to_array(img)
    img_2d = np.expand_dims(img_array, axis=0)
    img_processed = preprocess_input(img_2d)
    img_feature = img_model.predict(img_processed, verbose=0)

    # Flatten the image to get embdedding
    img_embd = img_feature.flatten()

    return img_embd


def create_embd_database(folder_path):
    embd_database = {}  # creating a dictionary for embd_database
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):

            # Files are either pdf or docx
            if filename.endswith((".pdf", ".docx")):
                try:
                    text = extract_text(file_path)
                    embedding = model.encode(text, normalize_embeddings=True)
                    embd_database[filename] = embedding

                except FileNotFoundError:
                    st.info(f"Error: File '{filename}' not found. Hence skipping...")

                except Exception as e:
                    st.info(f"Error moving '{filename}': {e}")

            # If files are images
            elif filename.endswith(img_ext):
                try:
                    # Generate image embedding
                    img_embd = img_embedding(file_path)
                    embd_database[filename] = img_embd

                except FileNotFoundError:
                    st.info(f"Error: File '{filename}' not found. Hence skipping...")

                except Exception as e:
                    st.info(f"Error moving '{filename}': {e}")

            else:
                pass

    return embd_database


def sha_algo():
    st.info('\nSHA model is running....\n')

    create_folder(duplicates_folder)  # Create a duplicate folder if doen't exists
    # load existing hash database or create a new one
    if os.path.exists("hash_database.txt"):
        with open("hash_database.txt", "r") as f:
            hash_database = eval(f.read())
    else:
        hash_database = create_hash_database(document_folder)
        with open("hash_database.txt", "w") as f:
            f.write(str(hash_database))

    # Check for new documents
    for filename in os.listdir(document_folder):
        file_path = os.path.join(document_folder, filename)
        if os.path.isfile(file_path):
            # New document: update hash database and potentially move existing duplicates
            try:
                with open(file_path, "rb") as f:
                    md5_hash = hashlib.md5(f.read()).hexdigest()
                    sha256_hash = hashlib.sha256(f.read()).hexdigest()
                hash_database[filename] = (md5_hash, sha256_hash)

                # check for existing duplicates based on MD5 or SHA-256
                for existing_filename, existing_hashes in hash_database.items():
                    if (existing_filename != filename) and (md5_hash == existing_hashes[0]):  # checking for MD5
                        # move existing duplicates
                        move_duplicate(existing_filename, document_folder, duplicates_folder, filename)
                        break  # only move one duplicate per new document
            except FileNotFoundError:
                st.info(f"Error: File'{filename}' not found. Skipping...")

    with open("hash_database.txt", "w") as f:
        f.write(str(hash_database))
    st.info("Hash database updated.")


def llm_algo():
    st.info('\nLLM model is running....\n')

    create_folder(duplicates_folder)  # Create a duplicate folder if doen't exists

    # Setting up the model bge-large-zh-v1.5
    if os.path.exists("embd_database.pkl"):
        with open("embd_database.pkl", "rb") as file:
            embd_database = pickle.load(file)
    else:
        st.info("Creating Database")
        embd_database = create_embd_database(
            document_folder)  # Duplicate folder path is provided because at the databasse if any duplicates are there the function will move the files to duplicates folder
        with open("embd_database.pkl", "wb") as file:
            pickle.dump(embd_database, file)
        st.info("Embedding database updated.")

    for filename in os.listdir(document_folder):
        file_path = os.path.join(document_folder, filename)
        if os.path.isfile(file_path):

            # Check for new pdf or docx files
            if filename.endswith((".pdf", ".docx")):
                # New document: update hash database and potentially move existing duplicates
                try:
                    text = extract_text(file_path)
                    embedding = model.encode(text, normalize_embeddings=True)
                    embd_database[filename] = embedding
                    # check for existing duplicates based on embedding

                    for existing_filename, existing_embd in embd_database.items():
                        # checking for Duplicates
                        if existing_filename.endswith((".pdf", ".docx")) and (existing_filename != filename):
                            similarity_score = existing_embd @ embedding.T

                            if similarity_score > cosine_score:
                                move_duplicate_llm(existing_filename, filename, similarity_score, document_folder,
                                                   duplicates_folder)
                                break  # only move one duplicate per new document


                except FileNotFoundError:
                    st.info(f"Error: File'{filename}' not found. Skipping...")

                except Exception as e:
                    st.info(f"Error moving '{filename}': {e}")

            # Check for image files
            elif filename.endswith(img_ext):

                try:
                    # Generate image embedding
                    img_embd = img_embedding(file_path)
                    embd_database[filename] = img_embd

                    for file_db, db_img_embd in embd_database.items():

                        if file_db.endswith(img_ext) and (file_db != filename):
                            similiarity_score_img = cosine_similarity_img(img_embd, db_img_embd)

                            if similiarity_score_img > img_similarity_score:  # Check with the image threshold limit(cosine score)
                                move_duplicate_llm(file_db, filename, similiarity_score_img, document_folder,
                                                   duplicates_folder)
                                break


                except FileNotFoundError:
                    st.info(f"Error: File'{filename}' not found. Skipping...")

                except Exception as e:
                    st.info(f"Error moving '{filename}': {e}")

            else:
                pass

    # Save the embedding dictionary inn pkl
    with open("embd_database.pkl", "wb") as file:
        pickle.dump(embd_database, file)
    st.info("LLM database completed")


def main():
    global document_folder
    global duplicates_folder

    # Define directory paths using streamlit
    document_folder = st.text_input('Input folder path where documents are stored')
    duplicates_folder = st.text_input('Input folder path where duplicate documents file to be stored')
    if document_folder == None or duplicates_folder == None:
        st.text("Enter the folders path")
    if document_folder != "":
        if os.path.isdir(document_folder):
            st.text(f"Selected document folder: {document_folder}")
        else:
            st.text("No path exists! \nEnter the documents folder again")
    if duplicates_folder != "":
        if os.path.isdir(document_folder):
            st.text(f"Selected duplicate document folder: {document_folder}")
        else:
            st.text("No path exists! \nEnter the duplicate documents folder again")

    # document_folder = r"C:\Users\ankit\Downloads\Project Duplicate\Testing docs"
    # duplicates_folder = r"C:\Users\ankit\Downloads\Project Duplicate\Testing docs\Duplicate_files"
    # sha_algo()
    # llm_algo()

    if os.path.isdir(document_folder) and os.path.isdir(duplicates_folder) and st.button("Move Duplicate files"):
        sha_algo()
        llm_algo()
        st.success("\n Completed!")


if __name__ == "__main__":
    main()
