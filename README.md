# "Semantic Text Similarity and Entailment Analysis"
***Project stored as = "Semantic_Similarity_TFIDF_BERT(pooled).ipynb"***

**Objective:** The project aims to find semantically similar sentences in a dataset to a given search query. It employs both TF-IDF vectorization and BERT embeddings to calculate similarity scores.

**Key responsibilities:**

1. **Data Loading:** The project begins by loading a dataset from a file, specifically the SNLI dataset, which contains sentence pairs and labels.

2. **Search Query:** A search query, which is a sample sentence, is provided as the basis for similarity comparisons with sentences in the dataset.

3. **TF-IDF Vectorization:** TF-IDF vectorization is used to convert the text data into numerical representations. The TF-IDF vectors are computed for sentence pairs.

4. **BERT Embeddings:** BERT (Bidirectional Encoder Representations from Transformers) embeddings are extracted for the same sentence pairs. These embeddings capture the contextual information of the sentences.

5. **Similarity Calculation:** Cosine similarity and Euclidean distance metrics are used to compute the similarity between the search query and each sentence in the dataset. A weighted average similarity score is also calculated, combining the two metrics.

6. **Top Similar Sentences:** The project identifies the top 10 most similar sentences to the search query based on the calculated similarity scores.

7. **Display Similarity Report:** The project displays a report for the top similar sentences, including the sentences themselves, their labels (e.g., 'entailment', contradiction), cosine similarity, Euclidean distance, and the weighted similarity score.

**Data Processing in Chunks:** To avoid running out of memory, the dataset is processed in smaller chunks, making it more memory-efficient.

Overall, the project showcases how to combine traditional TF-IDF techniques with state-of-the-art BERT embeddings to compute semantic similarity and classify textual entailment tasks. It's a practical demonstration of utilizing powerful natural language processing tools to solve real-world tasks involving text data.

# "Semantic Textual Similarity Predictor (STSP) - BERT(Pooled)"
***Project stored as = "SemanticSimilarity_SNLI_BERT.ipynb"***

**Project Objective:**

The main objective of the project is to develop a robust natural language processing model that predicts the semantic similarity between pairs of sentences using the Stanford Natural Language Inference (SNLI) corpus. The project encompasses various stages, including data preprocessing, model building, and evaluation.

**Key resposibilities :**

1. **Data Preparation:**

--> **Data Loading:** The project starts by loading the SNLI corpus, containing pairs of sentences and their labels.
--> **Data Preprocessing:**
--------> **Check for Null Values:** Ensure data integrity by identifying and handling any missing values.
--------> **Filter Empty Labels:** Remove records with empty labels, retaining only sentences with valid labels.

2. **Label Encoding:**
The labels, including "contradiction," "entailment," and "neutral," are one-hot encoded into numerical format (0, 1, and 2).

3. **Data Generator Creation:**
A custom data generator is developed to efficiently process the data, incorporating sentence pairs, labels, batch size, and the BERT tokenizer for tokenization.

4. **Model Building:**
A BERT-based (Pooled) model is constructed to predict semantic similarity between sentence pairs.

5. **Model Training:**
The model is trained using both training and validation data concurrently, optimizing training efficiency by activating use_multiprocessing = True.
-------> Training Accuracy: 0.934
-------> Training Loss: 0.196
-------> Validation Accuracy: 0.849
-------> Validation Loss: 0.495

6. **Model Evaluation:**
The model's performance is assessed using test data to evaluate its ability to predict semantic similarity.
--------> Test Accuracy: 0.8407
--------> Test Loss: 0.5092
7. **Inference:**
Custom sentences are provided to the model to predict similarity scores and labels.

In summary, the model is performing reasonably well, especially in terms of training accuracy. However, the slight decrease in accuracy on validation and test data suggests that there might be room for fine-tuning and optimization to enhance generalization. The model can be considered a good starting point, and further improvements can be explored based on specific project requirements and performance expectations.
