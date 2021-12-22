# evaluating_topic-modeling_for_sujet_and_theme
This repository includes data and code for a paper on validating topic modeling as an approximation to sujet, fabula, and theme.

The data folder contains the relevant processed corpus representations:

The following two large files are given access by the following link:
mallet output file: https://www.dropbox.com/s/o83eh6frd8b5eay/chunks_novellas.mallet?dl=0
document term matrix for corpus (basis for baseline classification): https://www.dropbox.com/s/1xu002wdd3xy2jd/doc_term_matrix_5000mfw_lemmatized_tfidf.csv?dl=0

All other files are stored in the data folder in this repository, including
a) Two tables represt the metadata: 
- Historical metadata for genre and year of publication in novellas.metadata.csv
- Annotations of the relevant sujets and themes in novellas_annotations.csv

b) Lists of words approximating sujets (in the wordlists_embeddings sub-folder) based on a method that is documented in the scripts folder: The wordlists can be generated with the generate_Document-Themes-Matrix.py script, which uses functions from the preprocessing.themes module.



c) The shares of topics for all documents (separated in chunks of 1000= words as output_composition_100topics.txt file (mallet output)
The topic keys for our 100 topics model. 

Our topic model was generated with mallet 2.0.8 in a process of recursive adaption of preprocessing and modeling parameters, such as:
- chunk length: 1000 words
- reduction on lemmatized nouns, verbs, adjectivs, and adverbs before chunking
- removing proper names and stop words
- parameter settings in mallet: alpha: 2.5, beta: 0.05, optimize-burn-in: 200, optimize-interval: 100
- the concret mallet-command: `mallet train-topics --input chunks_fixed.mallet --num-topics 100 --optimize-burn-in 200 --optimize-interval 100 --output-doc-topics output_composition_100topics1.txt --output-topic-keys output_keys_100topics1.txt --num-threads 2 --beta 0.005 --alpha 2.5`


The code with the scripts is stored in the scripts folder, relevant functions and classes are in the presetting module.

The relevant embedding based lists can be generated with the generate_sujet_wordlists_based_on_embedding.py.
The functions that use word embedding and a SpaCy model are in the preprocessing.presetting module.

The corpus representation as a matrix of documents and the share of themes can be generated with generate_Document-Themes-Matrix.py.
The relevant classes and functions are in the preprocessing.themes and the .corpus modules. 
This process requires a full corpus of texts, which can, due to copyright issues and due to the policy of a larger project, not be provided. The resulting Document-Themes-Matrix.csv is stored in the data folder.

The validations in section 4 of the paper can be reproduced with: 
extensional_topic_validation.py,
extensional_word-embeddings-lists-validation.py,
baseline_classification_BOW-model.py

Figure 2 in the conclusion can be reproduced with visualize_sujet_development.py


