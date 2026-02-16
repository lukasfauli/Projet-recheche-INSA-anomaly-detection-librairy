 #function to omit NAn values

import pandas as pd
def drop_duplicate(df) -> pd.DataFrame:

    '''
    Remove all the duclicates from the dataframe df
    print a message withe the number of row removed, if there is no row removed there is no message
    :param df: Description
    :return: Description
    :rtype: DataFrame
    '''
    df_cleaned=df.drop_duplicates().copy()
    rows_removed = len(df) - len(df_cleaned)
    if rows_removed > 0:
        print(f"INFO: {rows_removed} lines have been removed.")
    elif rows_removed == 0:
        print("INFO: No duplicate lines have been found.")
    return df_cleaned

def drop_NAn(df)-> pd.DataFrame:

    '''
    remove all the lines containing a Nan from the dataframe df
    print a message withe the number of row removed, if there is no row removed there is no message
    :param df: the studied data frame 
    :return: the cleaned data_frame
    :rtype: DataFrame
    '''

    df_cleaned = df.dropna().copy()
    rows_removed = len(df) - len(df_cleaned)
    if rows_removed > 0:
        print(f"INFO: {rows_removed} lines have been removed.")
    elif rows_removed == 0:
        print("INFO: No lines with NaN values have been found.")    
    return df_cleaned

def convert_datetime(df)->pd.DataFrame:

    '''
    Convert the column datetime of a dataframe to type datetime64[ns]
    :param df: Description
    :return: Description
    :rtype: DataFrame
    '''
    if "datetime" in df.columns:
        df_cleaned=df.copy()
        df_cleaned.datetime=pd.to_datetime(df_cleaned.datetime)
        return df_cleaned
    else:
        raise KeyError("the datetime column doesn't exist ")


def convert_dict_to_dataframe(dictionnary)->pd.DataFrame:

    '''
    Convert the data from dictionnary to dataframe
    :param dict: dictionnary
    :return: a dataframe
    :rtype: DataFrame
    '''
    return pd.DataFrame(dictionnary)

def cleaning_dataframe(df)->pd.DataFrame:

    '''
    Combinaison of all the previous function in one 
    to call only one function to do the whole preprocessing
    :param df: the dictionnary obtain after the data load
    :return: a clean data frame ready to be use
    :rtype: DataFrame
    '''
    df_cleaned = df.copy()
    drop_duplicate(df_cleaned)
    drop_NAn(df_cleaned)
    convert_datetime(df_cleaned)

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
        cleaned_df = cleaning_dataframe(df)
        cleaned_sessions[session_name] = cleaned_df
        
    return cleaned_sessions