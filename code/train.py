import tensorflow as tf
import numpy as np
import reader as data_reader
from model import create_model, load_model_weights
from aspect_extractor_callback import AspectExtractorCallBack
import argparse
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, required=True,
                    help="The path to the input directory")
parser.add_argument("--out_dir", type=str, required=True,
                    help="The path to the output directory")
parser.add_argument("--hidden_size", type=int, default=200,
                    help="Embeddings dimension (default=200)")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size (default=64)")
parser.add_argument("--vocab_size", type=int, default=9000,
                    help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("--aspect_num", type=int, default=14,
                    help="The number of aspects specified by users (default=14)")
parser.add_argument("--epochs", type=int, default=15,
                    help="Number of epochs (default=15)")
parser.add_argument("--neg_num", type=int, default=20,
                    help="Number of negative instances (default=20)")
parser.add_argument("--max_seq_len", type=int, default=128,
                    help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--domain", type=str, default='restaurant',
                    help="domain of the corpus {restaurant, beer, ...}")

args = parser.parse_args()

args.in_dir = args.in_dir + "/" + args.domain
args.out_dir = args.out_dir + "/" + args.domain
args.emb_path = args.in_dir + "/w2v_embedding"
args.model_path = args.out_dir + "/model"


def data_generator(data, neg_num):
    i = 0
    data_len = len(data)
    np.random.shuffle(data)
    while True:
        if i == data_len:
            np.random.shuffle(data)
            i = 0
        pos_sample = data[i]
        neg_indices = np.random.choice(data_len, neg_num)
        neg_samples = data[neg_indices]
        yield pos_sample, neg_samples, 1
        i += 1


def format_data(pos, neg, label):
    return (pos, neg), label


def get_tf_dataset(file_path, vocab):
    train_x, _ = data_reader.read_dataset(file_path,
                                          vocab,
                                          args.max_seq_len)
    train_x = tf.keras.preprocessing.sequence.pad_sequences(
        train_x,
        maxlen=args.max_seq_len,
        padding='post')

    ds = tf.data.Dataset.from_generator(
        data_generator,
        args=[train_x, args.neg_num],
        output_signature=(
            tf.TensorSpec(shape=(args.max_seq_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(args.neg_num, args.max_seq_len), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    ds = ds.map(format_data).batch(args.batch_size)
    logger.info('Number of training examples: %d' % len(train_x))
    return ds


def main():
    vocab = data_reader.create_vocab(args.in_dir + "/train.txt")
    vocab_inv = {i: w for w, i in vocab.items()}
    args.vocab_size = len(vocab)
    logger.info('Length of vocab: %d' % len(vocab))

    train_ds = get_tf_dataset(args.in_dir + "/train.txt", vocab)

    model = create_model(args)
    # run predict to trigger model building
    model.predict(train_ds.take(1))
    model = load_model_weights(model, args, vocab)
    model.get_layer('embedding_layer').trainable = False

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.out_dir + "/weights_{loss:.4f}",
            monitor="loss",
            save_best_only=True,
            mode="min"
        ),
        AspectExtractorCallBack(
            filepath=args.out_dir + "/aspect.log",
            vocab_inv=vocab_inv,
            word_per_aspect=100
        )
    ]

    model.fit(train_ds,
              steps_per_epoch=100,
              epochs=args.epochs,
              callbacks=callbacks,
              verbose=1,
              )


if __name__ == "__main__":
    main()
