# function to omit NAn values

import pandas as pd


def gen_vprint(verbose):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    return vprint


vprint = gen_vprint(verbose=True)


def drop_duplicate(df, inplace=True) -> pd.DataFrame:
    '''
    Remove all the duclicates from the dataframe df
    print a message withe the number of row removed, if there is no row removed there is no message
    :param df: Description
    :return: Description
    :rtype: DataFrame
    '''
    len_before = len(df)
    df_cleaned = df.drop_duplicates(inplace=inplace)
    rows_removed = len_before - len(df_cleaned)

    # Print the number of rows removed, if any
    if rows_removed > 0:
        vprint(f"INFO: {rows_removed} lines have been removed.")
    elif rows_removed == 0:
        vprint("INFO: No duplicate lines have been found.")

    return df_cleaned


def drop_nan(df) -> pd.DataFrame:
    '''
    remove all the lines containing a Nan from the dataframe df
    print a message withe the number of row removed, if there is no row removed there is no message
    :param df: the studied data frame 
    :return: the cleaned data_frame
    :rtype: DataFrame
    '''
    # TODO: avoid data duplication for memory optimization
    df_cleaned = df.dropna().copy()
    rows_removed = len(df) - len(df_cleaned)
    if rows_removed > 0:
        print(f"INFO: {rows_removed} lines have been removed.")
    elif rows_removed == 0:
        print("INFO: No lines with NaN values have been found.")
    return df_cleaned


def convert_datetime(df, inplace=True, datetime_column="datetime") -> pd.DataFrame:
    '''
    Convert the column datetime of a dataframe to type datetime64[ns]
    :param df: Description
    :return: Description
    :rtype: DataFrame
    '''
    if not datetime_column in df.columns:
        raise KeyError("the datetime column doesn't exist ")
    
    df_cleaned = df if inplace else df.copy()
    df_cleaned[datetime_column] = pd.to_datetime(
        df_cleaned[datetime_column], errors='coerce')
    return df_cleaned


# def convert_dict_to_dataframe(dictionnary) -> pd.DataFrame:
#     '''
#     Convert the data from dictionnary to dataframe
#     :param dict: dictionnary
#     :return: a dataframe
#     :rtype: DataFrame
#     '''
#     return pd.DataFrame(dictionnary)


def clean_dataframe(df, inplace=True) -> pd.DataFrame:
    '''
    Combinaison of all the previous function in one 
    to call only one function to do the whole preprocessing
    :param df: the dictionnary obtain after the data load
    :return: a clean data frame ready to be use
    :rtype: DataFrame
    '''
    df_cleaned = df if inplace else df.copy()
    drop_duplicate(df_cleaned, inplace=inplace)
    drop_nan(df_cleaned, inplace=inplace)
    convert_datetime(df_cleaned, inplace=inplace)

    return df_cleaned


def preprocess_all_sessions(sessions_dict: dict) -> dict:
    """
    Apply the full preprocessing pipeline to multiple driving sessions.
    This function iterates through a dictionary of sessions, applying all the previous cleaning function
    to each session individually to maintain data 
    integrity and avoid cross-contamination between different runs.
    Args:
    Returns:
        dict: A dictionnary with the cleaned DataFrames.
    """
    cleaned_sessions = dict()
    for session_name, df in sessions_dict.items():
        print(f"Nettoyage de la session : {session_name}")
        cleaned_df = clean_dataframe(df)
        cleaned_sessions[session_name] = cleaned_df

    return cleaned_sessions
