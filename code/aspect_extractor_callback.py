import tensorflow as tf
import codecs
import numpy as np


class AspectExtractorCallBack(tf.keras.callbacks.Callback):
    """Extract aspect words when the loss in at is min
    Arguments:
        filepath: str, path to save aspect words
        vocab_inv: index to word
        word_per_aspect: number of words per aspect to save
        save_freq: Number of batch to wait. if 0, the callback run after each epoch
    """

    def __init__(self, filepath, vocab_inv, word_per_aspect=1, save_freq=0):
        super(AspectExtractorCallBack, self).__init__()
        self.filepath = filepath
        self.vocab_inv = vocab_inv
        self.word_per_aspect = word_per_aspect
        self.save_freq = save_freq
        self.word_emb = None
        self.aspect_emb = None

    def on_train_begin(self, logs=None):
        self.best = np.Inf

    def on_train_batch_end(self, batch, logs=None):
        if self.save_freq == 0:
            return
        if (batch + 1) % self.save_freq != 0:
            return
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.word_emb = self.model.embedding_layer.get_weights()[0]
            self.aspect_emb = self.model.decoder.aspect_to_sen_layer.get_weights()[0].T

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq != 0:
            return
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.word_emb = self.model.embedding_layer.get_weights()[0]
            self.aspect_emb = self.model.decoder.aspect_to_sen_layer.get_weights()[0]

    def on_train_end(self, logs=None):
        if (self.word_emb is None) or \
                (self.aspect_emb is None):
            return
        word_emb = tf.math.l2_normalize(self.word_emb, axis=-1)
        aspect_emb = tf.math.l2_normalize(self.aspect_emb, axis=-1)
        sims = tf.matmul(aspect_emb, word_emb, transpose_b=True)
        topk_idx = tf.math.top_k(sims, k=self.word_per_aspect).indices
        aspect_file = codecs.open(self.filepath, "w", "utf-8")
        for i in range(topk_idx.shape[0]):
            word_idx = np.array(topk_idx[i, :])
            words = [self.vocab_inv[indx] for indx in word_idx]
            aspect_file.write("Aspect %d:\n" % i)
            aspect_file.write(" ".join(words) + "\n\n")
        aspect_file.close()
