# Unsupervised Aspect Extraction
Codes with tensorflow2.4 for ACL2017 paper ‘‘An unsupervised neural attention model for aspect extraction’’. [(pdf)](http://aclweb.org/anthology/P/P17/P17-1036.pdf)

## Data
You can find the pre-processed datasets and the pre-trained word embeddings in [[Download]]. The zip file should be decompressed and put in the main folder.

You can also do preprocessing for you own original datasets. For preprocessing, put the file in the main folder and run 
```
python preprocess.py
python word2vec.py
```
respectively in code/ . The preprocessed files and trained word embeddings for each domain will be saved in a folder preprocessed_data/.

## Train
Under code/ and type the following command for training:
```
python train.py \
--in_dir ../preprocessed_data \
--out_dir ../output_dir
--domain $domain
```
or
```
sh run.sh
```

After training, two output files will be saved in ../output_dir/$domain/: 1) *aspect.log* contains extracted aspects with top 100 words for each of them. 2) *weights* contains the saved model weights

## Dependencies

python 3
* tensorflow 2.4
* numpy 1.18.1
* gensim 3.8.3
