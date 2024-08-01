# Project 3: Medicine Recommendation System

## Project Overview
This project aims to build a machine learning model that recommends medications based on patient inputs using advanced NLP techniques and transformer models. We utilized various machine learning libraries, natural language processing (NLP) techniques, TF-IDF vectorization, cosine similarity, neural network model and deep learning to provide medication recommendations. Additionally, sentiment analysis is performed on user reviews to understand the general perception of the recommended medications. The project includes data preprocessing, model training, evaluation, and visualization of the results.

## Project Goals
The goal is to recommend the most suitable medication for patients based on their symptoms, allergies, current medications, and other medical conditions. The system should be able to analyze the textual data provided by the patients and match it with the medication information to find the best fit. Additionally, the project includes sentiment analysis on user reviews to gauge the effectiveness and reception of medications.

## Project Steps
1. Data Preprocessing:
- Load and clean the dataset.
- Combine relevant text columns into a single column for processing.
- Vectorize the combined text using TF-IDF.

2. Model Training:
- Encode the target labels using LabelEncoder.
- Split the data into training and testing sets.
- Build and train a neural network model using TensorFlow and Keras.

3. Medication Recommendation:
- Collect patient data through user input.
- Preprocess and vectorize the patient data.
- Calculate cosine similarity between the patient data and medication data.
- Recommend the medication with the highest similarity score.

4. Sentiment Analysis:
- Perform sentiment analysis on user reviews using VADER sentiment analysis tool.
- Display the sentiment scores for each review.

5. Visualization:
- Generate various plots to visualize the data and model performance.

## Technologies Used
- **Python**: Programming language.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **Scikit-learn**: Machine learning library.
- **TensorFlow/Keras**: Deep learning framework.
- **SpaCy**: NLP library.
- **WordCloud**: Visualization of text data.
- **Seaborn/Matplotlib**: Data visualization.
- **VADER (vaderSentiment)**: Sentiment nalysis. 
- **Whisper**: Speech-to-text model from OpenAI.

## Data Source
The dataset used for this project is a collection of medication information, including names, uses, and other relevant details. The dataset was cleaned and preprocessed to make it suitable for model training.

## Conclusion
The initial implementation of the medication recommendation system did not perform as expected. The model struggled to achieve meaningful accuracy and the validation loss remained high throughout the training epochs. This indicates that the model was not able to effectively learn from the data to make accurate predictions.
