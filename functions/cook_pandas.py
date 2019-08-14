import pandas as pd
import copy

# name: dum_col()
# function: create dummy columns based on one categorical column in a dataframe, original and first dropped
# dependencies: 
#     import pandas as pd
#     import copy
# arguments:
#     df: dataframe
#     col_name: 'column_name'
#     prefix: dummy column names prefix
# by: Yuzhang Huang @ 2019.07.27
def dum_col(df,col_name,prefix):
    df_new = copy.deepcopy(df)
    df_temp = pd.get_dummies(df_new[col_name],prefix=prefix,drop_first=True)
    df_new = df_new.drop(columns=[col_name])
    df_new = pd.concat([df_new, df_temp], axis=1)
    return df_new