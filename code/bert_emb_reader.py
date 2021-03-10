import logging
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertEmbReader:

    def __init__(self, checkpoint_file, vocab_file):
        logger.info('Loading embeddings from: ' + checkpoint_file)
        self.embeddings = {}
        emb_matrix = []

        self.vocab = [line.strip() for line in open(vocab_file)]
        embedding = tf.train.load_variable(checkpoint_file, "bert/embeddings/word_embeddings")
        for i, word in enumerate(self.vocab):
            self.embeddings[word] = list(embedding[i])
            emb_matrix.append(list(embedding[i]))

        self.emb_dim = len(self.embeddings["nice"])
        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)

        logger.info('  #vectors: %i, #dimensions: %i' % (self.vector_size, self.emb_dim))

    def get_embedding(self, word):
        try:
            return self.embeddings[word]
        except KeyError:
            return None
    
    def get_word_matrix(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.items():
            try:
                emb_matrix[index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass
        emb_matrix = np.array(emb_matrix)

        logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100*counter/len(vocab)))
        # L2 normalization
        norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        return norm_emb_matrix

    def get_aspect_matrix(self, n_clusters):
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        clusters = km.cluster_centers_

        # L2 normalization
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
        return norm_aspect_matrix.astype(np.float32)
    
    def get_emb_dim(self):
        return self.emb_dim


if __name__ == "__main__":
    import sys
    reader = BertEmbReader(sys.argv[1], sys.argv[2])
    emb1 = np.array(reader.get_embedding("我")).reshape((1, -1))
    distances = tf.matmul(emb1, reader.emb_matrix, transpose_b=True)
    near_ids = tf.math.top_k(distances, 100).indices[0, :]
    for i in near_ids:
        print(reader.vocab[i])
