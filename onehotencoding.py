import pandas as pd


# functions for one_hot_encoding and one_hot_decoding
def one_hot_encoding(df: pd.DataFrame):
    cate_name = df.columns.to_numpy()
    cate_class_number = []
    cate_class = []
    for i in range(df.columns.shape[0]):
        cate_class.append(df.iloc[:, i].unique())
        cate_class_number.append(df.iloc[:, i].nunique())

    for i in range(df.columns.shape[0]):
        df = pd.concat([df, pd.get_dummies(df[cate_name[i]], prefix=cate_name[i])], axis=1)
        df = df.drop(columns=cate_name[i])

    return cate_name, cate_class_number, cate_class, df


def one_hot_decoding(df: pd.DataFrame, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df