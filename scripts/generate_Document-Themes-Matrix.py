system = "wcph113"

# This script for generating a matrix of theme shares based on sujet word lists requires a corpus of texts. Due to copyright
# issues and the process of an onging larger project, the corpus that was used for the paper cannot be provided here.
# The output file DocThemesMatrix.csv is provided in the data folder of this repository.

import os
from preprocessing.presetting import global_corpus_directory, vocab_lists_dicts_directory, load_stoplist, language_model_path, global_corpus_representation_directory
from preprocessing.themes import DocThemesMatrix

corpus_path = global_corpus_directory(system) # here the copurs path of text files has to be specified

my_model_de = language_model_path(system)



list_of_wordlists = []

for filepath in os.listdir(vocab_lists_dicts_directory(system)):
    if "_wordlist.txt" in filepath:
        wordlist = load_stoplist(os.path.join(vocab_lists_dicts_directory(system), filepath))
        list_of_wordlists.append(wordlist)


print(list_of_wordlists)

normalization_table_path = os.path.join(vocab_lists_dicts_directory(system), "normalization_table.txt")


matrix_obj = DocThemesMatrix(corpus_path= corpus_path, list_of_wordlists=list_of_wordlists,
                             list_eliminate_pos_tags=["SYM", "PUNCT", "NUM", "SPACE"], list_of_pos_tags=None, keep_pos_items=False,
                             language_model=my_model_de, remove_hyphen=True, normalize_orthogr=True, normalization_table_path=normalization_table_path,
                             correct_ocr=True, eliminate_pagecounts=True, handle_special_characters=True,
                             inverse_translate_umlaute=False,
                             eliminate_pos_items=True,
                             lemmatize=True,
                             sz_to_ss=False, translate_umlaute=False,
                             remove_stopwords=False)
print(matrix_obj.data_matrix_df)

outfile_path = os.path.join(global_corpus_representation_directory(system), "DocThemesMatrix.csv")
matrix_obj.save_csv(outfile_path)



