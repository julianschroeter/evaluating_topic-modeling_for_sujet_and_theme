import numpy as np
import pandas as pd
import os
from preprocessing.presetting import global_corpus_representation_directory
import re

def years_to_periods(input_df, category_name,start_year, end_year, epoch_length, new_periods_column_name):
    """

    :param input_df:
    :param category_name: Column name for year of publication, for example "Jahr_ED"
    :param list_of_epochal_thresholds: for example [1700, 1730, 1760, 1790, 1820, 1850, 1880, 1910, 1940, 1970, 2000]
    :param new_periods_column_name: the name of the column with new periods category
    :return: a new dataframe with a new column new_periods_column_name
    """
    input_df[new_periods_column_name] = np.nan
    list_of_year_tuples_as_periods = []
    x = start_year

    while x <= end_year:
        tuple = (x, x+epoch_length)
        list_of_year_tuples_as_periods.append(tuple)
        x = x+ epoch_length

    for tuple in list_of_year_tuples_as_periods:
        start, end = tuple

        input_df[str(tuple)] = input_df[category_name].apply(lambda x: str(str(start)+"-"+str(end)) if start <= x < end else np.nan)
        input_df[new_periods_column_name] = input_df[new_periods_column_name].combine_first(input_df[str(tuple)])
        input_df.drop(columns=[str(tuple)], inplace=True)
    input_df[new_periods_column_name] = input_df[new_periods_column_name].replace(np.nan, 0)
    return input_df



def full_genre_labels(df, replace_dict= {"Gattungslabel_ED_normalisiert": {"N": "Novelle", "E": "Erzählung", "0E": "Prosaerzählung ohne Label",
                                    "R": "Roman", "M": "Märchen", "XE": "Prosaerzählung mit anderem Label"}}):
    """
    Hard coded function to replace genre abbreviation in meta data table with full genre labels
    :param df: metadata tabel with category/column "Gattungslabel_ED"
    :return: data frame with full genre names for N, E, 0E, R, M, XE
    """
    df.replace(replace_dict, inplace=True)
    return df



def standardize_meta_data_medium(df, medium_column_name):
    df[medium_column_name] = df[medium_column_name].astype("category")
    df['medium'] = df[medium_column_name].apply(lambda x: 'urania' if 'Urania' in x else(
                                                            'westermanns monatshefte' if 'Westermanns' in x else(
                                                            'gartenlaube' if 'Gartenlaube' in x else(
                                                             'daheim' if 'Daheim' in x else(
                                                             'Urania' if 'urania' in x else(
                                                             'dtrundsch' if 'Deutsche Rundschau' in x else(
                                                             'werke' if 'Werke' in x else(
                                                              'aglaja' if 'aglaja' in x else(
                                                              'aglaja' if 'Aglaja' in x else(
                                                              'neue rundschau' if 'neue Rundschau' in x else(
                                                              'tb_fuer_damen' if 'Taschenbuch für Damen' in x else(
                                                              'sonst_tb' if 'TB' in x else(
                                                              'jahrbuch' if 'Jahrbuch' in x else(
                                                              'zeitung' if 'NeuePresseWien' in x else(
                                                              'westmon' if 'WestMon' in x else(
                                                              'cotta' if 'Cottas' in x else(
                                                              'journal' if 'journal' in x else(
                                                              'anthologie' if 'Anthol' in x else(
                                                              'anthologie' if 'Alm_Novellenkranz' in x else(
                                                              'roman' if 'Roman' in x else(
                                                              'grimm_khm' if 'grimm_khm' in x else(
                                                              'iris' if 'Iris' in x else(
                                                              'buch' if 'Buch' in x else(
                                                              'pantheon' if 'Pantheon' in x else(
                                                              'dtnovsch' if 'DTNovSchatz' in x else(
                                                              'kalender' if 'Kalender' in x else(
                                                              'zyklus' if 'Zyklus' in x else 'other')))))))))))))))))))))))))))

    df['medium_type'] = df['medium'].apply(lambda x: 'tb' if (x == 'aglaja' or x == 'urania' or x == "tbvielliebchen" or x == 'tb' or x == 'iris' or x == 'kalender') else(
                        'anthologie' if (x == 'anthologie' or x == "almanachnovellenkranz" or x == 'grimm_khm' or x == 'werke') else(
                            'novsamml' if (x == 'dtnovsch' or x == 'pantheon') else(
                                 'famblatt' if (x == 'gartenlaube' or x == 'daheim') else(
                                    'rundschau' if (x == 'dtrundsch' or x == 'westmon' or x == 'neuerundschau') else(
                                       'journal' if (x == 'journal' or x == 'cotta' or x == 'neuepressewien') else(
                                               'anthologie' if x == 'zyklus' else(
                                                   'buch' if (x == 'roman' or x == 'buch') else 'No_journal'))))))))

    return df

def extract_file_ids_period(df=None, from_year= 0,to_year=2020, year_column_name="Jahr_ED"):
    """

    :param df:
    :param from_year: start year is integer
    :param to_year: last year (included) as integer
    :return: a list of file ids for texts that fall within the selected period
    """
    df["within_period"] = df[year_column_name].apply(lambda x: 1 if from_year <= x <= to_year else 0)
    df_reduced = df[df["within_period"] == 1]
    return df_reduced.index.tolist()

def extract_ids_categorical(df=None, metadata_category="Gattungslabel_ED", variables_list=["N","E","0E"]):
    """

    :param df:
    :param metadata_category:
    :param variables_list:
    :return: a list of file ids for texts that possess one of the categorical feature in the metadata_category
    """
    values_dict = {metadata_category: variables_list}
    df = df[df.isin(values_dict).any(1)]
    return df.index.tolist()


def generate_media_dependend_genres(df):
    """

    :param df:
    :param media:
    :return:
    """
    famblatt_df = df[df["medium_type"] == "famblatt"]

    famblatt_df["dependent_genre"] = df["Gattungslabel_ED"].apply(lambda x: "Familienblatt_Novelle" if x == "N" else ("Familienblatt_Erzählung" if x == "E" else
                                                                                                                      ("Famlienblatt_sonst_Erzählung" if x == "0E" else "familienblatt_other")))

    rundschau_df = df[df["medium_type"] == "rundschau"]

    rundschau_df["dependent_genre"] = df["Gattungslabel_ED"].apply(lambda x: "Rundschau_Novelle" if x == "N" else ("Rundschau_Erzählung" if x == "E" else
            ("Rundschau_sonst_Erzählung" if x == "0E" else "other")))
    tb_df = df[df["medium_type"] == "tb"]
    tb_df["dependent_genre"] = df["Gattungslabel_ED"].apply(lambda  x: "tb_Novelle" if x == "N" else
    ("tb_Erzählung" if x == "E" else ("tb_sonst_Erzählung" if x == "0E" else "tb_other")))

    anthol_df = df[df["medium_type"] == "anthologie"]
    anthol_df["dependent_genre"] = df["Gattungslabel_ED"].apply(lambda x:"anthologie_Novelle" if x == "N" else
    ("anthologie_Erzählung" if x == "E" else ("anthologie_sonst_Erzählung" if x == "0E" else "anthologie_other")))

    journal_df = df[df["medium_type"] == "journal"]
    journal_df["dependent_genre"] = df["Gattungslabel_ED"].apply(lambda x:"journal_Novelle" if x == "N" else
    ("journal_Erzählung" if x == "E" else ("journal_sonst_Erzählung" if x == "0E" else "journal_other")))
    buch_df = df[df["medium_type"] == "buch"]
    buch_df["dependent_genre"] = df["Gattungslabel_ED"].apply(lambda x: "buch_Novelle" if x == "N" else
    ("buch_Erzählung" if x == "E" else ("buch_sonst_Erzählung" if x == "0E" else "buch_other" )))
    frames = pd.concat([famblatt_df, rundschau_df, tb_df, anthol_df, journal_df, buch_df])
    print(frames)
    return frames

def category_counts_per_periods(dep_var, values_list, input_df, periods_category):
    """
    returns a new data frame which counts the number of items of a variable, such as genre
    """

