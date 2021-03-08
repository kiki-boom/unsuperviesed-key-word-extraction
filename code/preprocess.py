from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str,
                    help="input dir path", required=True)
parser.add_argument("-o", "--output_dir", type=str,
                    help="output dir path", required=True)
args = parser.parse_args()


def parse_sentence(line):
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem


def preprocess_train(domain):
    f = codecs.open(args.input_dir + '/' + domain + '/train.txt', 'r', 'utf-8')
    out = codecs.open(args.output_dir + '/' + domain + '/train.txt', 'w', 'utf-8')

    for line in f:
        tokens = parse_sentence(line)
        if len(tokens) > 0:
            out.write(' '.join(tokens)+'\n')


def preprocess_test(domain):
    # For restaurant domain, only keep sentences with single 
    # aspect label that in {Food, Staff, Ambience}

    f1 = codecs.open(args.input_dir + '/' + domain + '/test.txt', 'r', 'utf-8')
    f2 = codecs.open(args.input_dir + '/' + domain + '/test_label.txt', 'r', 'utf-8')
    out1 = codecs.open(args.output_dir + '/' + domain + '/test.txt', 'w', 'utf-8')
    out2 = codecs.open(args.output_dir + '/' + domain + '/test_label.txt', 'w', 'utf-8')

    for text, label in zip(f1, f2):
        label = label.strip()
        if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
            continue
        tokens = parse_sentence(text)
        if len(tokens) > 0:
            out1.write(' '.join(tokens) + '\n')
            out2.write(label+'\n')


def preprocess(domain):
    logger.info("%s train set ...", domain)
    preprocess_train(domain)
    logger.info("%s test set ...", domain)
    preprocess_test(domain)


if __name__ == "__main__":
    logger('Preprocessing raw review sentences ...')
    preprocess('restaurant')
    preprocess('beer')
