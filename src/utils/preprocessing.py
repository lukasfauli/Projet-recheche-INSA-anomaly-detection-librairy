

import pandas as pd


def gen_vprint(verbose):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    return vprint


vprint = gen_vprint(verbose=True)


def drop_duplicate(df, inplace=True,Verbose=False) -> pd.DataFrame:
    '''
    Remove all the duclicates from the dataframe df
    print a message withe the number of row removed, if there is no row removed there is no message
    :param df: Description
    :return: Description
    :rtype: DataFrame
    '''
    len_before = len(df)
    df.drop_duplicates(inplace=inplace)
    rows_removed = len_before - len(df)

    # Print the number of rows removed, if any. False by default
    if Verbose:
        if rows_removed > 0:
            vprint(f"INFO: {rows_removed} lines have been removed.")
        elif rows_removed == 0:
            vprint("INFO: No duplicate lines have been found.")

    return df


def drop_nan(df,inplace=True,Verbose=False) -> pd.DataFrame:
    '''
    remove all the lines containing a Nan from the dataframe df
    print a message withe the number of row removed, if there is no row removed there is no message
    :param df: the studied data frame 
    :return: the cleaned data_frame
    :rtype: DataFrame
    '''

    len_before=len(df)
    df.dropna(inplace=inplace)
    rows_removed = len_before - len(df)

    # Print the number of rows removed, if any. False by default
    if Verbose:
        if rows_removed > 0:
            print(f"INFO: {rows_removed} lines have been removed.")
        elif rows_removed == 0:
            print("INFO: No lines with NaN values have been found.")
    return df


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

def renorm_time(df, datetime_column="datetime") -> pd.DataFrame:
    '''
    transform the datetime into time starting from 0 to help to plot the columns
    it creates a new column at the end of the data frame
    :param df: Description
    :return: Description
    :rtype: DataFrame
    '''
    if not datetime_column in df.columns:
        raise KeyError("the datetime column doesn't exist ")
    times=df[datetime_column]
    first_time=df[datetime_column].iloc[0]
    df['time_real'] = (times - first_time).dt.total_seconds()
    return df


def convert_dict_to_dataframe(dictionnary) -> pd.DataFrame:
     '''
     Convert the data from dictionnary to dataframe
     :param dict: dictionnary
     :return: a dataframe
     :rtype: DataFrame
     '''
     return pd.DataFrame(dictionnary)

def clean_dataframe(df, inplace=True) -> pd.DataFrame:
    '''
    Combinaison of all the previous function in one 
    to call only one function to do the whole preprocessing
    :param df: the dictionnary obtain after the data load
    :return: a clean data frame ready to be use
    :rtype: DataFrame
    '''
    df_cleaned = df if inplace else df.copy()
    convert_datetime(df_cleaned, inplace=inplace)
    renorm_time(df, datetime_column="datetime")
    drop_duplicate(df_cleaned, inplace=inplace)
    drop_nan(df_cleaned, inplace=inplace)
    
    

    return df_cleaned



