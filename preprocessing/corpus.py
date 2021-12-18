import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from preprocessing.text import Text
from copy import deepcopy


"""
2. Generate word lists from corpus vocabulary pased on POS-Tagging (with spaCy), for example: function word lists, semantic word lists, noun lists, verb lists, adjective lists etc.
"""

''' 3. Generating, editing, manipulating, and transforming a Corpus Representation as term document matrix:  
The following scripts organize the preprocessing steps of:
1. a) Correcting common and domain specific OCR errors, based on practical experience with reading OCR; and b) Normalization if necessary (transforming Ä,Ö,Ü,ß to ae, oe, ue, ss for example for topic modeling with mallet
2. Lemmatization (with spacy)
3. POS-Tagging, (with spacy)
4. Selection of POS-Types
5. Stopword-Handling
6. Chunking
'''


class DocFeatureMatrix():
    """
    abstract parent class for TDM (= term document matrix) and TopicDocMatrix as child classes. Feature vectors for each document are represented as row vectors.
    """
    def __init__(self, data_matrix_filepath, metadata_csv_filepath=None, data_matrix_df=None, metadata_df=None, mallet=False):
        self.data_matrix_df = data_matrix_df
        self.data_matrix_filepath = data_matrix_filepath
        self.metadata_csv_filepath = metadata_csv_filepath
        self.metadata_df = metadata_df
        self.mallet = mallet

        print("data_csv_filepath is:", self.data_matrix_filepath)
        print("metadata_csv_filepath is:", self.metadata_csv_filepath)


        if self.metadata_csv_filepath is not None:
            self.metadata_df = pd.read_csv(self.metadata_csv_filepath, index_col=0)

        if self.data_matrix_filepath is not None:

            if self.mallet == False:
                self.data_matrix_df = pd.read_csv(self.data_matrix_filepath, index_col=0)

            elif self.mallet == True:
                print( "hier muss die Anpassung an mallet output df erfolgen: an doc-topic-matrix")
                df = pd.read_csv(self.data_matrix_filepath, index_col=1, sep='\t', header=None)
                df.drop(df.columns[0], axis=1, inplace=True)
                #df.columns = [i for i in range(len(df.columns))]
                df.reset_index(inplace=True)
                self.data_matrix_df = df



    def reduce_to(self, reduction_list, return_eliminated_terms_list=False):
        """
        Based on the vocabulary of matrix_df, a new FeatureDoc_matrix object is returned only with features which are listed in reduction_list.
        if flag return_eliminated_terms_list == True, also the list of eliminated terms is returned
        if flag return_eliminated_terms_list == True, only a reduced DTM instance is returned.
        """
        object = deepcopy(self)
        columns_list = list(object.data_matrix_df.columns.values)
        terms_to_reduce = list(filter(lambda x: x in reduction_list, columns_list))
        eliminated_terms_list = list(filter(lambda x: x not in reduction_list, columns_list))
        object.data_matrix_df = object.data_matrix_df.loc[:, terms_to_reduce]
        if return_eliminated_terms_list == False:
            return object
        else:
           return object,  eliminated_terms_list

    def eliminate(self, elimination_list):
        """
        Based on the vocabulary of matrix_df, a new FeatureDoc_matrix object is returned. All
        features listed in elimination list are eliminated from the data_matrix_df.
        Typical use cases are stopword lists and name lists.

        """
        object = deepcopy(self)
        columns_list = list(object.data_matrix_df.columns.values)
        terms_to_drop = list(filter(lambda x: x in elimination_list, columns_list))
        object.data_matrix_df.drop(terms_to_drop, axis=1, inplace=True)

        return object


    def add_metadata(self, metadata_category):
        """
        adds metadata from metadata table to the data_matrix_df attribute in new object and overwrites this attribute with the enriched matrix.
        :param metadata_category: Type of metadata as specified as column name in the meta data table (from csv)
        :return: a copy of the object with data_matrix_df attribute is generated. This attribute is a data frame and supplied with the respective metadata category as an additional column
        """
        object = deepcopy(self)
        object.data_matrix_df = object.data_matrix_df.join(self.metadata_df[metadata_category])
        return object

    def reduce_to_categories(self, metadata_category, label_list):
        values_dict = {metadata_category: label_list}
        object = deepcopy(self)
        object.data_matrix_df = object.data_matrix_df[object.data_matrix_df.isin(values_dict).any(1)]
        return object

    def save_csv(self, outfile_path):
        """
        save feature document matrix in the data_matrix_df attribute as csv file under outfile_path
        :param outfile_path: filepath to store the file.
        """
        self.data_matrix_df.to_csv(path_or_buf=outfile_path)



