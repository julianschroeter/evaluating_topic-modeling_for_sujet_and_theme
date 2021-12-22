# This script performs the extensional validation for several relevant topics.
# for each list and for each relevant sujet or theme, a t-test and a boot-strapped classification task (with 10000 re-samplings and iterations) is performed.

# import own modules
from preprocessing.topic_postprocessing import DocTopicMatrix
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, local_temp_directory, global_corpus_representation_directory, load_stoplist
from preprocessing.corpus import DocFeatureMatrix
from preprocessing.classification import all_validate

# import official python libraries
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats

# specify system for filepaths:
system_name = "wcph113"

# set variables and filepaths
data_matrix_filepath = os.path.join(local_temp_directory(system_name), "output_composition_100topics.txt")
language_model = language_model_path(system_name)
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system_name), "Bibliographie.csv")
df = pd.read_csv(os.path.join(global_corpus_representation_directory(system_name), "Bibliographie.csv"), index_col=0)

colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system_name), "my_colors.txt"))
textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system_name), "textanalytic_metadata.csv")
topic_doc_matrix = DocTopicMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = textanalytic_metadata_filepath, mallet=True)

# calculate mean topic shares over all chunks for each document:
topic_doc_matrix = topic_doc_matrix.adjust_doc_chunk_multiindex()
topic_doc_matrix.data_matrix_df.to_csv(os.path.join(global_corpus_representation_directory(system_name), "test-doc-topics.csv"))
df_all_segments = topic_doc_matrix.data_matrix_df
matrix = topic_doc_matrix.mean_doclevel()
doc_topic_df = matrix.data_matrix_df

doc_topic_df = doc_topic_df.head(-7) # as there are 7 files in the original data with wrong ids at the end of the matrix, the last 7 rows are removed from the data frame.


#a matrix object is created that can be combined with metadata
matrix = DocFeatureMatrix(data_matrix_filepath=None, data_matrix_df = doc_topic_df,
                         metadata_df=None,metadata_csv_filepath=textanalytic_metadata_filepath,
                        mallet=False)


print('calculate results for T-Test and classification for topic 34 and romantic love:')
red_matrix = matrix.reduce_to([34]) # = candidate romantic love topic

# add annotation metadata (= ("is lovestory"))
red_matrix = red_matrix.add_metadata("liebesspannung")
# transform the system of four values in the original annotations to binary values:
red_matrix.data_matrix_df.replace({"liebesspannung": {"gering": "no",
                                                        "mittel": "yes",
                                                        "stark": "yes", "kaum": "no", "nein" : "no"
                                          }}, inplace=True)

red_matrix.data_matrix_df["liebesspannung"] = red_matrix.data_matrix_df["liebesspannung"].fillna("other", inplace=False)
df = red_matrix.data_matrix_df
scaler = StandardScaler()
# scale the data:
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())
# generate two contrary samples:
category = "liebesspannung" # "stadt_land" # "region"
first_value = "yes" # "land" # "rom"
second_value = "no" # "None" #"non_rom"

sample_df = df[df[category] == first_value]
counter_sample_df = df[df[category] == second_value]

# customized bootstrapping method for classification with 1000 repititions of resampling:
print("bootstrapped validation of log reg (mean accuracy and std): ", all_validate(sample_df, counter_sample_df, 10000)[:2])

# prepare samples for t-test: remove category labels
sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: t-statistics, p: ", F, p)

print("mean shares of relevant topic for each sample:")
grouped_df = df.groupby([category]).mean()
print(grouped_df)

print("Size of samples:")
grouped_df = df.groupby([category]).count()
print(grouped_df)

# next topic and theme
print('calculate results for T-Test and classification for topic 36 and romantic love:')
red_matrix = matrix.reduce_to([36]) # = candidate romantic love topic
# add annotation metadata (= ("is lovestory"))
red_matrix = red_matrix.add_metadata("liebesspannung")
# transform the system of four values in the original annotations to binary values:
red_matrix.data_matrix_df.replace({"liebesspannung": {"gering": "no",
                                                        "mittel": "yes",
                                                        "stark": "yes", "kaum": "no", "nein" : "no"
                                          }}, inplace=True)

red_matrix.data_matrix_df["liebesspannung"] = red_matrix.data_matrix_df["liebesspannung"].fillna("other", inplace=False)
df = red_matrix.data_matrix_df
scaler = StandardScaler()
# scale the data:
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

# generate two contrary samples:
category = "liebesspannung" # "stadt_land" # "region"
first_value = "yes" # "land" # "rom"
second_value = "no" # "None" #"non_rom"

sample_df = df[df[category] == first_value]
counter_sample_df = df[df[category] == second_value]

# customized bootstrapping method for classification with 1000 repititions of resampling:
print("bootstrapped validation of log reg (mean accuracy and std): ", all_validate(sample_df, counter_sample_df, 10000)[:2])

# prepare samples for t-test: remove category labels
sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: t-statistics, p: ", F, p)


print('calculate results for T-Test and classification for topic 47 and romantic love:')
red_matrix = matrix.reduce_to([47]) # = candidate romantic love topic
# add annotation metadata (= ("is lovestory"))
red_matrix = red_matrix.add_metadata("liebesspannung")
# transform the system of four values in the original annotations to binary values:
red_matrix.data_matrix_df.replace({"liebesspannung": {"gering": "no",
                                                        "mittel": "yes",
                                                        "stark": "yes", "kaum": "no", "nein" : "no"
                                          }}, inplace=True)
red_matrix.data_matrix_df["liebesspannung"] = red_matrix.data_matrix_df["liebesspannung"].fillna("other", inplace=False)
df = red_matrix.data_matrix_df
scaler = StandardScaler()
# scale the data:
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())
# generate two contrary samples:
category = "liebesspannung" # "stadt_land" # "region"
first_value = "yes" # "land" # "rom"
second_value = "no" # "None" #"non_rom"
sample_df = df[df[category] == first_value]
counter_sample_df = df[df[category] == second_value]

# customized bootstrapping method for classification with 1000 repititions of resampling:
print("bootstrapped validation of log reg (mean accuracy and std): ", all_validate(sample_df, counter_sample_df, 10000)[:2])

# prepare samples for t-test: remove category labels
sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: t-statistics, p: ", F, p)


print('calculate results for T-Test and classification for topic 64 and rural surrounding:')
red_matrix = matrix.reduce_to([64]) # = candidate

# add annotation metadata (= ("is lovestory"))
red_matrix = red_matrix.add_metadata("stadt_land")


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
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

# generate two contrary samples:
category = "stadt_land" # "region"
first_value = "land" # "rom"
second_value = "None" #"non_rom"

sample_df = df[df[category] == first_value]
counter_sample_df = df[df[category] == second_value]

# customized bootstrapping method for classification with 1000 repititions of resampling:
print("bootstrapped validation of log reg (mean accuracy and std): ", all_validate(sample_df, counter_sample_df, 10000)[:2])

# prepare samples for t-test: remove category labels
sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: t-statistics, p: ", F, p)

print("mean shares of relevant topic for each sample:")
grouped_df = df.groupby([category]).mean()
print(grouped_df)

print("Size of samples:")
grouped_df = df.groupby([category]).count()
print("grouped std: ")
print(grouped_df)


print('calculate results for T-Test and classification for topic 28 and rural surrounding:')
red_matrix = matrix.reduce_to([28]) # = candidate

# add annotation metadata (= ("is lovestory"))
red_matrix = red_matrix.add_metadata("stadt_land")


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
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

# generate two contrary samples:
category = "stadt_land" # "region"
first_value = "land" # "rom"
second_value = "None" #"non_rom"

sample_df = df[df[category] == first_value]
counter_sample_df = df[df[category] == second_value]

# customized bootstrapping method for classification with 1000 repititions of resampling:
print("bootstrapped validation of log reg (mean accuracy and std): ", all_validate(sample_df, counter_sample_df, 10000)[:2])

# prepare samples for t-test: remove category labels
sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: t-statistics, p: ", F, p)


print('calculate results for T-Test and classification for topic 38 and rural surrounding:')
red_matrix = matrix.reduce_to([38]) # = candidate

# add annotation metadata (= ("is lovestory"))
red_matrix = red_matrix.add_metadata("stadt_land")


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
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())

# generate two contrary samples:
category = "stadt_land" # "region"
first_value = "land" # "rom"
second_value = "None" #"non_rom"

sample_df = df[df[category] == first_value]
counter_sample_df = df[df[category] == second_value]

# customized bootstrapping method for classification with 1000 repititions of resampling:
print("bootstrapped validation of log reg (mean accuracy and std): ", all_validate(sample_df, counter_sample_df, 10000)[:2])

# prepare samples for t-test: remove category labels
sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: t-statistics, p: ", F, p)


