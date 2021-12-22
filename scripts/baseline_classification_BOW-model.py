# This script performs the extensional validation for classificaton for the annotated sets based on a baseline of a features set of 5000 most frequent words, tf-idf normalized.
# for each list and for each relevant sujet or theme, a t-test and a boot-strapped classification task (with 10000 re-samplings and iterations) is performed.

# import own modules
from preprocessing.presetting import  vocab_lists_dicts_directory, global_corpus_representation_directory, load_stoplist, global_corpus_raw_dtm_directory
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.classification import all_validate

# import official python libraries
import os
from sklearn.preprocessing import StandardScaler
from scipy import stats

# specify system for filepaths:
system_name = "wcph113"

# set variables and filepaths
data_matrix_filepath = os.path.join(global_corpus_raw_dtm_directory(system_name), "raw_dtm_5000mfw_lemmatized_l2_5000mfw.csv")
colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system_name), "my_colors.txt"))
textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system_name), "textanalytic_metadata.csv")

# generate matrix object with metadata
matrix = DocFeatureMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = textanalytic_metadata_filepath, mallet=False)


print('calculate results for T-Test and classification for annotated sets of romantic love:')


# add annotation metadata (= ("is lovestory"))
red_matrix = matrix.add_metadata("liebesspannung")
# transform the system of four values in the original annotations to binary values:
red_matrix.data_matrix_df.replace({"liebesspannung": {"gering": "no",
                                                        "mittel": "yes",
                                                        "stark": "yes", "kaum": "no", "nein" : "no"
                                          }}, inplace=True)

red_matrix.data_matrix_df["liebesspannung"] = red_matrix.data_matrix_df["liebesspannung"].fillna("other", inplace=False)
df = red_matrix.data_matrix_df
scaler = StandardScaler()
# scale the data:
#df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())
# generate two contrary samples:
category = "liebesspannung" # "stadt_land" # "region"
first_value = "yes" # "land" # "rom"
second_value = "no" # "None" #"non_rom"

sample_df = df[df[category] == first_value]
counter_sample_df = df[df[category] == second_value]

# customized bootstrapping method for classification with 1000 repititions of resampling:
print("bootstrapped validation of log reg (mean accuracy and std): ", all_validate(sample_df, counter_sample_df, 1000)[:2])

# prepare samples for t-test: remove category labels
sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: t-statistics, p: ", F, p)


print('calculate results for T-Test and classification for annotated sets of rural surrounding:')


# add annotation metadata (= ("is lovestory"))
red_matrix = matrix.add_metadata("stadt_land")


# transform the system of values in the original annotations to binary values:
red_matrix.data_matrix_df.replace({"stadt_land": {"Bergdorf": "land",
                                                        "land": "land",
                                                        "Alpenregion": "land",
                                              "stadt" :"None", "stadt+land": "None"
                                          }}, inplace=True)


red_matrix.data_matrix_df["stadt_land"] = red_matrix.data_matrix_df["stadt_land"].fillna("other", inplace=False)
df = red_matrix.data_matrix_df

scaler = StandardScaler()
# scale the data:
#df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

# generate two contrary samples:
category = "stadt_land" # "region"
first_value = "land" # "rom"
second_value = "None" #"non_rom"

sample_df = df[df[category] == first_value]
counter_sample_df = df[df[category] == second_value]

# customized bootstrapping method for classification with 1000 repititions of resampling:
print("bootstrapped validation of log reg (mean accuracy and std): ", all_validate(sample_df, counter_sample_df, 1000)[:2])

# prepare samples for t-test: remove category labels
sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: t-statistics, p: ", F, p)


print('calculate results for T-Test and classification for annotated sets of Romanesque environment:')
red_matrix = matrix.add_metadata("region")
red_matrix.data_matrix_df.replace({"region": {"Italien": "rom",
                                                        "Spanien": "rom",
                                                        "Frankreich": "rom", "Lateinamerika" : "rom",
                                          "Karibik" : "rom", "Chile" : "rom", "Portugal":"rom",
                                          "Deutschland" : "non_rom", "Österreich" : "non_rom",
                                          "Niederlande" : "non_rom", "Ungarn" : "non_rom", "Russlan" : "non_rom",
                                          "Polen" : "non_rom", "Schweden" : "non_rom", "Universum" : "non_rom", "Dänemark" :"non_rom",
                                          "Russland" : "non_rom", "Schweiz" : "non_rom", "Karibik" : "rom", "Alpen" : "non_rom",
                                          "Nordamerika" :"non_rom", "Meer":"non_rom"

                                          }}, inplace=True)
red_matrix.data_matrix_df["region"] = red_matrix.data_matrix_df["region"].fillna("other", inplace=False)
df = red_matrix.data_matrix_df
scaler = StandardScaler()
# scale the data:
#df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())
# generate two contrary samples:
category =  "region"
first_value = "rom"
second_value = "non_rom"

sample_df = df[df[category] == first_value]
counter_sample_df = df[df[category] == second_value]

# customized bootstrapping method for classification with 1000 repititions of resampling:
print("bootstrapped validation of log reg (mean accuracy and std): ", all_validate(sample_df, counter_sample_df, 1000)[:2])

# prepare samples for t-test: remove category labels
sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: t-statistics, p: ", F, p)
