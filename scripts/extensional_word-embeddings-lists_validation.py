# This script performs the extensional validation for the custom-made word lists based on embeddings.
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

#set the name of the system
system_name = "wcph113" # "my_mac" # "wcph104"

data_matrix_filepath = os.path.join(global_corpus_representation_directory(system_name), "DocThemesMatrix.csv")
language_model = language_model_path(system_name)
metadata_csv_filepath = os.path.join(global_corpus_representation_directory(system_name=system_name), "Bibliographie.csv")
df = pd.read_csv(os.path.join(global_corpus_representation_directory(system_name), "Bibliographie.csv"), index_col=0)

colors_list = load_stoplist(os.path.join(vocab_lists_dicts_directory(system_name), "my_colors.txt"))

textanalytic_metadata_filepath = os.path.join(global_corpus_representation_directory(system_name), "textanalytic_metadata.csv")

matrix = DocFeatureMatrix(data_matrix_filepath= data_matrix_filepath, data_matrix_df=None, metadata_df=None,
                                  metadata_csv_filepath = textanalytic_metadata_filepath, mallet=False)



print('calculate results for T-Test and classification for word embedding based list for rural surrounding and the annotation of rural surrounding:')
red_matrix = matrix.reduce_to(["Feld"]) # = candidate romantic love topic

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

print("mean share for each sample:")
grouped_df = df.groupby([category]).mean()
print(grouped_df)

print("Size of samples:")
grouped_df = df.groupby([category]).count()
print("grouped std: ")
print(grouped_df)



print('calculate results for T-Test and classification for word embedding based list for romantic love and the annotation of romantic love:')
red_matrix = matrix.reduce_to(["lieben.1"]) # = candidate romantic love topic



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

print("mean share of each sample:")
grouped_df = df.groupby([category]).mean()
print(grouped_df)

print("Size of samples:")
grouped_df = df.groupby([category]).count()
print(grouped_df)


print('calculate results for T-Test and classification for embedding based list of Romanesque environment and the annotation of romanesque environment:')
red_matrix = matrix.reduce_to(["Marseille"]) # = candidate word list

red_matrix = red_matrix.add_metadata("region")
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
df.iloc[:, :1] = scaler.fit_transform(df.iloc[:, :1].to_numpy())
# generate two contrary samples:
category =  "region"
first_value = "rom"
second_value = "non_rom"

sample_df = df[df[category] == first_value]
counter_sample_df = df[df[category] == second_value]

# customized bootstrapping method for classification with 1000 repititions of resampling:
print("bootstrapped validation of log reg (mean accuracy and std): ", all_validate(sample_df, counter_sample_df, 10000)[:2])

# prepare samples for t-test: remove category labels
sample_df = sample_df.drop([category], axis=1)
counter_sample_df = counter_sample_df.drop([category], axis=1)
F, p = stats.ttest_ind(sample_df, counter_sample_df)
print("T-Test ind two samples: t-statistics, p: ", F, p)

print("mean share of each sample:")
grouped_df = df.groupby([category]).mean()
print(grouped_df)

print("Size of samples:")
grouped_df = df.groupby([category]).count()
print(grouped_df)
