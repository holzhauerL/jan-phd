import numpy as np      # linear algebra
import pandas as pd     # data processing

class EmbeddingAnalyzer():
    """
    Class to analyze the embeddings of sentences and topics.
    """
    def __init__(self, embeddings, sentences, topics=[], verbose=True):
        """
        Initialize the EmbeddingAnalyzer with the embeddings.

        Parameters:
        embeddings (dict): The embeddings dictionary.
        sentences (list): The list of sentences.
        topics (list): The list of topics. Defaults to an empty list.
        verbose (bool): Whether to print the outputs. Defaults to True.
        """
        self.embeddings = embeddings
        self.sentences = sentences
        self.topics = topics

        self.df_sentences = pd.DataFrame([(sentence, embeddings.get(sentence)) for sentence in sentences], columns=['Sentence', 'Embedding'])
        self.df_topics = pd.DataFrame([(topic, embeddings.get(topic)) for topic in topics], columns=['Topic', 'Embedding'])

        self.S = np.vstack(self.df_sentences['Embedding'])  # (n_sentences, embedding_dim)
        self.T = np.vstack(self.df_topics['Embedding'])     # (n_topics, embedding_dim)

        self.verbose = verbose

    def cosine_similarity_matrix(self, S=None, T=None, epsilon=1e-10):
        """
        Compute the cosine similarity matrix between two embedding matrices S and T.

        Parameters:
        S (numpy.ndarray): The sentence embedding matrix of shape (n_sentences, embedding_dim).
        T (numpy.ndarray): The topic embedding matrix of shape (n_topics, embedding_dim).
        epsilon (float): A small epsilon to avoid division by zero.

        Returns:
        numpy.ndarray: A matrix of shape (n_sentences, n_topics) where each entry (i, j)
                    represents the cosine similarity between the i-th sentence and the j-th topic.
        """
        if S is None:
            S = self.S
        if T is None:
            T = self.T
        # Normalize each row in A
        S_norm = S / (np.linalg.norm(S, axis=1, keepdims=True) + epsilon)

        # Normalize each row in B
        T_norm = T / (np.linalg.norm(T, axis=1, keepdims=True) + epsilon)

        # Compute the cosine similarity matrix
        cosine_similarity = np.dot(S_norm, T_norm.T)

        if self.verbose:
            print(f"Shape of sentence embeddings:       {S.shape}")
            print(f"Shape of topic embeddings:          {T.shape}")
            print(f"Shape of cosine similarity matrix:  {cosine_similarity.shape}")
        
        return cosine_similarity