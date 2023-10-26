# "Semantic Text Similarity and Entailment Analysis"

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

# 
