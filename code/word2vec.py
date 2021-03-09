import gensim
import codecs
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", type=str,
                    default="../preprocessed_data/train.txt",
                    help="input file path")
parser.add_argument("-o", "--output_dir", type=str,
                    default="../preprocessed_data",
                    help="output dir path")
args = parser.parse_args()


class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main():
    source = args.input_file
    model_file = args.output_dir + "/w2v_embedding"
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=10, workers=4)
    model.save(model_file)


if __name__ == "__main__":
    main()



