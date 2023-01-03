import pandas as pd


# define three normalization methods
# define norm and denorm function
def max_abs_norm(data: pd.DataFrame, column: str):
    max_val = data[column].abs().max()
    data[column] = data[column] / max_val
    return data, max_val


def min_max_norm(data: pd.DataFrame, column: str):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column] = (data[column] - min_val) / (max_val - min_val)
    return data, min_val, max_val


def standardization(data: pd.DataFrame, column: str):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column] = (data[column] - mean_val) / std_val
    return data, mean_val, std_val


def max_abs_denorm(data: pd.DataFrame, column: str, norm_dict: dict):
    max_val = norm_dict[column]
    data[column] = data[column] * max_val
    return data


def min_max_denorm(data: pd.DataFrame, column: str, norm_dict: dict):
    min_val = norm_dict[column][0]
    max_val = norm_dict[column][1]
    data[column] = data[column] * (max_val - min_val) + min_val
    return data


def destandardization(data: pd.DataFrame, column: str, norm_dict: dict):
    mean_val = norm_dict[column][0]
    std_val = norm_dict[column][1]
    data[column] = data[column] * std_val + mean_val
    return data


def norm(data: pd.DataFrame, columns: [], norm_types: []):
    norm_dict = {}
    for i in range(len(columns)):
        if norm_types[i] == 'max_abs':
            data, max_val = max_abs_norm(data, columns[i])
            norm_dict.update({columns[i]: max_val})

        if norm_types[i] == 'min_max':
            data, min_val, max_val = min_max_norm(data, columns[i])
            norm_dict.update({columns[i]: [min_val, max_val]})

        if norm_types[i] == 'standard':
            data, mean_val, std_val = standardization(data, columns[i])
            norm_dict.update({columns[i]: [mean_val, std_val]})

    return data, norm_dict


def denorm(data: pd.DataFrame, columns: [], norm_types: [], norm_dict: dict):
    for i in range(len(columns)):
        if norm_types[i] == 'max_abs':
            data = max_abs_denorm(data, columns[i], norm_dict)

        if norm_types[i] == 'min_max':
            data = min_max_denorm(data, columns[i], norm_dict)

        if norm_types[i] == 'standard':
            data = destandardization(data, columns[i], norm_dict)

    return data