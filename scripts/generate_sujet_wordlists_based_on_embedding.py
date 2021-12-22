# Imports
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, save_stoplist, keywords_to_semantic_fields
import os

system = "wcph113"

my_language_model_de = language_model_path(system)

# this script generates lists of similar words based on the following procedure:
# 1. Basedn on priar assumptions, language knowledge, and exploration of topics, a list of exptected input words
# for a specific sujet, fabula, or a concept within a thematic claim (such as 'romantik love'
# is generated
# 2. Then, the function keywords_to_semantic_fields() from the preprocessing.presetting module generates
# for each word the list of the 50 most similar words according to word embeddings of a Spacy language model (large German)
# 3. All similar words for all words of the input_words are added, and
# 4. the 30 most frequent word types are returned as words indicating the relevant sujet.
# 5. The resulting lists are saved with the most frequent word in the filename.


# for the sujet of a rural setting:
input_words = ["Dorf", "Hof", "Knecht", "Feld", "Wald", "Förster", "Pflug", "Bauernhof", "Stall", "Pferd", "Kuh", "Landwirtschaft", "Bauer",
               "melken", "Acker", "Dorfkirche", "Dorfplatz", "bäuerlich", "Landwirt"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

filename = str(str(semantic_words_list[0])+"_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

# for the sujet of Romanesque setting:
input_words = ["Italien", "Frankreich", "Spanien", "italienisch", "französisch", "spanisch",
               "Nordfrankreich", "Südfrankreich", "Provence", "Paris", "Lyon", "Marseille", "Bordeaux",
                "Neapel", "Sizilien", "Toscana", "Florenz", "Venedig", "Rom", "Genua", "Madrid", "Salamanca"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

romaniawordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), "romania_wordlist.txt")
save_stoplist(semantic_words_list, romaniawordlist_filepath)

# for neutral love concept:
input_words = ["Liebe", "lieben", "verliebt", "Zuneigung", "Verehrung", "verehren"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), "Liebe_wordlist.txt")
save_stoplist(semantic_words_list, wordlist_filepath)

# for seafaring:

input_words = ["Meer", "Schiff", "Wind", "Galeere", "Segel", "Welle", "Sturm", "Mast", "Ufer", "Bord", "Insel", "Kahn", "Matrose", "Küste"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

filename = str(str(semantic_words_list[0])+"_wordlist.txt")
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

# for marriage:

input_words = ["Hochzeit", "Heirat", "Braut", "Bräutigam", "Brautvater", "Trauung", "Glück", "Ehe", "Eheglück"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

# for amour passion (according to Luhmann):
input_words = ["Liebe", "lieben", "verliebt", "Leidenschaft", "rasend", "Raserei", "verrückt", "Verrücktheit", "Täuschung", "täuschen", "Eifersucht", "Ehebruch",
                 "besessen", "Besessenheit", "Betrug", "betrügen", "Spiel", "Spielerei", "Liebschaft"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

filename = str(str(semantic_words_list[0])+"_amourpassion_wordlist.txt")
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)

# for fin amour:
input_words = ["Liebe", "lieben", "verliebt", "ideal", "schön", "Schönheit", "tadellos", "perfekt", "wunderschön"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

filename = str(str(semantic_words_list[0])+"_finamour_wordlist.txt")
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)


# for marriage live:
input_words = ["Liebe", "lieben", "verliebt", "Ehe", "Partnerschaft", "Heirat", "Harmonie", "Ehestreit", "Vernunft", "Freundschaft", "Kameradschaft", "Begleiter", "Partner", "Tochter", "Sohn", "Kind", "Haushalt"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))
filename = str(str(semantic_words_list[0])+"_ehe_wordlist.txt")
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)


# for individualized love (Luhmann's idea of romantic love):
input_words = ["Liebe", "lieben", "verliebt", "einzigartig", "einmalig", "einzig"]
semantic_words_list = keywords_to_semantic_fields(list_of_keywords=input_words, n_most_relevant_words=30,
                                                  spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))

filename = str(str(semantic_words_list[0])+"_romantliebe_wordlist.txt")
print(filename)
wordlist_filepath = os.path.join(vocab_lists_dicts_directory(system), filename)
save_stoplist(semantic_words_list, wordlist_filepath)