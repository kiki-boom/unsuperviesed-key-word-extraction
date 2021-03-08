import tensorflow as tf
from my_layers import Average, AttentionEncoder, Decoder, MaxMargin
from optimizers import get_optimizer
from w2v_emb_reader import W2VEmbReader
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class MyModel(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 aspect_num,
                 hidden_size=768,
                 dropout_rate=0.1,
                 **kwargs
                 ):
        super(MyModel, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.aspect_num = aspect_num
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.embedding_layer = tf.keras.layers.Embedding(
            self.vocab_size,
            self.hidden_size,
            mask_zero=True,
            name="embedding_layer"
        )
        self.average_layer = Average(name="average_layer")
        self.attention_encoder = AttentionEncoder(name="attention_encoder")
        self.decoder = Decoder(
            self.aspect_num,
            self.hidden_size,
            self.dropout_rate,
            name="decoder"
        )
        self.predict_layer = MaxMargin(name="loss_layer")

    def call(self, inputs, training=None):
        pos_sen = inputs[0]
        neg_sens = inputs[1]

        # positive sentence representation
        pos_word_emb = self.embedding_layer(pos_sen)
        pos_avg_sen_emb = self.average_layer(pos_word_emb)
        pos_att_sen_emb = self.attention_encoder(inputs=[pos_word_emb, pos_avg_sen_emb])

        # positive sentence reconstruction
        pos_rec_sen_emb = self.decoder(pos_att_sen_emb)

        # negative sentence embedding
        neg_word_emb = self.embedding_layer(neg_sens)
        neg_avg_sen_emb = self.average_layer(neg_word_emb)

        # loss
        pred = self.predict_layer(inputs=(pos_att_sen_emb, pos_rec_sen_emb, neg_avg_sen_emb))
        return pred


def max_margin_loss(y_true, y_pred):
    return tf.math.reduce_mean(y_pred)


def create_model(args):
    model = MyModel(
        vocab_size=args.vocab_size,
        aspect_num=args.aspect_num,
        hidden_size=args.hidden_size,
    )
    model.compile(
        #optimizer=get_optimizer(args),
        optimizer=tf.optimizers.Adam(learning_rate=5e-5),
        loss=max_margin_loss,
        metrics=[max_margin_loss],
    )
    return model


def load_model_weights(model, args, vocab):
    if args.emb_path:
        logger.info("init embeddings from %s" % args.emb_path)
        emb_reader = W2VEmbReader(args.emb_path)
        model.embedding_layer.embeddings.assign(
            emb_reader.get_word_matrix(
                vocab, model.embedding_layer.get_weights()[0]))
        aspect_embedding_matrix = emb_reader.get_aspect_matrix(args.aspect_num)
        model.decoder.aspect_to_sen_layer.kernel.assign(aspect_embedding_matrix)

    latest_checkpoint = tf.train.latest_checkpoint(args.model_path)
    if latest_checkpoint:
        logger.info("Initial model from %s" % latest_checkpoint)
        model.load_weights(latest_checkpoint)
    return model
