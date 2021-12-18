# evaluating_topic-modeling_for_sujet_and_theme
This repository includes data and code for a paper on validating topic modeling as an approximation to sujet, fabula, and theme.

The data folder contains the relevant processed corpus representations:

Two tables represting metadata: 
Historical metadata for genre and year of publication in novellas.metadata.csv
Annotations of the relevant sujets and themes in novellas_annotations.csv

Lists of words approximating sujets (in the wordlists_embeddings sub-folder) based on a method that is documented in the code folder: The wordlists can be generated with the generate_Document-Themes-Matrix.py script, which uses functions from the preprocessing.themes module.

The shares 

The shares of topics for all documents (separated in chunks of 1000= words as output_composition_100topics.txt file (mallet output)
The topic keys for our 100 topics model. 

Our topic model was generated with mallet 2.0.8 in a process of recursive adaption of preprocessing and modeling parameters, such as:
- chunk length: 1000 words
- reduction on nouns, verbs, adjectivs, and adverbs before chunking
- removing proper names and stop words
- hyperparamter optimization in mallet: ...
