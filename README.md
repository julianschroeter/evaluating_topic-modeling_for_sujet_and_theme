# evaluating_topic-modeling_for_sujet_and_theme
This repository includes data and code for a paper on validating topic modeling as an approximation to sujet, fabula, and theme.

The data folder contains the relevant processed corpus representations:

Two tables represt the metadata: 
- Historical metadata for genre and year of publication in novellas.metadata.csv
- Annotations of the relevant sujets and themes in novellas_annotations.csv

Lists of words approximating sujets (in the wordlists_embeddings sub-folder) based on a method that is documented in the scripts folder: The wordlists can be generated with the generate_Document-Themes-Matrix.py script, which uses functions from the preprocessing.themes module.



The shares 

The shares of topics for all documents (separated in chunks of 1000= words as output_composition_100topics.txt file (mallet output)
The topic keys for our 100 topics model. 

Our topic model was generated with mallet 2.0.8 in a process of recursive adaption of preprocessing and modeling parameters, such as:
- chunk length: 1000 words
- reduction on lemmatized nouns, verbs, adjectivs, and adverbs before chunking
- removing proper names and stop words
- hyperparamter optimization in mallet: ...


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


