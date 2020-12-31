import pandas as pd
from sklearn import preprocessing


differential_features = ['drug_1.0', 'drug_2.0', 'drug_3.0', 'drug_4.0', 'drug_5.0', 'drug_7.0', 'drug_8.0']


def get_differential_values_counts(df, differential_features):
    # returns an array with the number of samples from each (one hot encoded) feature from a lost
    return [len(df[df[feature] == 1]) for feature in differential_features]


# the function gets a data frame, a list of one-hot encoded features and  a thresould and returns a filtered data
# frame without values that are rare
def exclude_negligible_differential_features(df, differential_features, minumum_number_of_samples=100):
    counts = get_differential_values_counts(df, differential_features)
    to_keep = list(differential_features)
    for i, c in enumerate(counts):
        if (c < minumum_number_of_samples):
            df = df[df[differential_features[i]] != 1]
            df = df.drop([differential_features[i]], axis=1)
            to_keep.remove(differential_features[i])
    return df, to_keep


def normalize_features_in_df(df, features, cofficient=3):
    x = df[features].values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = (min_max_scaler.fit_transform(
        x) * cofficient).astype(
        int)  # todo generalize, check the effect of the multipaction and esxtract to a different fuction
    df[features] = pd.DataFrame(x_scaled)
    return df


def get_data(train_path, test_path):
    pd_train = pd.read_csv(train_path)
    pd_test = pd.read_csv(test_path)
    df_all = (pd.concat([pd_train, pd_test])).drop(columns={'subjectkey'})
    # normalize the qids total variable (since it has a wide range )
    df_all = normalize_features_in_df(df_all, ["qstot"])
    # only consider differential features that are common
    df_all, to_keep = exclude_negligible_differential_features(df_all, differential_features)
    return df_all, len(pd_train), len(pd_test), to_keep
