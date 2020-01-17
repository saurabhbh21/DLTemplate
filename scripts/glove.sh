mkdir pretrained_model
cd pretrained_model

wget http://nlp.stanford.edu/data/glove.6B.zip

unzip glove.6B.zip
ls|grep -v "glove.6B.300d.txt"|xargs rm