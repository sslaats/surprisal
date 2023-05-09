# What's surprising about surprisal
## Code for the simulations in Slaats & Martin (2023). 
Supplementary materials: https://osf.io/xp3r7/

Script descriptions are: [script name] : [description]

### general scripts [all simulations]
* utils.py :  some basic functions, 
* predictor.py : class for the LSTM model
* vocabulary.py : class to keep track of indices & vocabulary items

### toy grammar [syntax leads to surprisal, surprisal obscures the view]
* grammar.py : specifies the toy grammar (phrase-structure rules)
* simulate-corpus.py : uses the grammar to generate n sentences for training of the LSTM model
* train-model.py : train model on toy grammar
* train-random-model.py : train model on scrambled output of toy grammar
* test-model.py : test model trained on toy grammar
* compare-models.py : compares the surprisal values on the test set between scrambled and structured models 
* language.csv : the vocabulary & POS to use for simulate-corpus.py

### natural language / OpenSubtitles  [syntax leads to surprisal, surprisal does not lead to syntax]
* preprocessing-opensubtitles.py : sentence & word tokenization and interpunction removal of OpenSubtitles corpus
* train-model-natural-1layer.py : train model on OpenSubtitles corpus
* test-model-natural.py : test model trained on OpenSubtitles corpus
* correlation-natural.py : compares the surprisal values on the test set between scrambled and structured models 
* clustering.py : use a RandomForestClassifier to classify surprisal values as coming from Spanish or English
