import pandas as pd
import numpy as np
from os.path import join
from typing import Tuple


def get_inliers_mask_target(df: pd.DataFrame) -> np.ndarray:
    # p = df.groupby('RanksID')['Perf_1Y'].min()
    p = df.groupby('RanksID')['Perf_1Y'].max()
    badids = set(p[abs(p) > TARGET_THRESHOLD].index)
    return np.array([el not in badids for el in df.RanksID])


def get_inliers_borders(arr: np.ndarray, lq: float = 0.025, rq: float = 0.975) -> Tuple[float, float]:
    q1, q2 = np.quantile(arr, [lq, rq])
    iqr = q2 - q1
    lb = q1 - 1.5 * iqr
    ub = q2 + 1.5 * iqr

    return lb, ub


def get_inliers_mask(arr: np.ndarray, lq: float = 0.025, rq: float = 0.975) -> np.ndarray:
    lb, ub = get_inliers_borders(arr, lq, rq)

    return (lb < arr) & (arr < ub)


data_path = 'data'
lq, rq = 0.05, 0.95
TARGET_THRESHOLD = 3

valid_columns = ['Date_round', 'RanksID', 'Perf_1Y', 'Country_of_Headquarters',
                 'Price_To_Tangible_Book_Value_Per_Share',
                 'C_Price_To_Tangible_Book_Value_Per_Share',
                 'Price_To_Book_Value_Per_Share', 'C_Price_To_Book_Value_Per_Share',
                 'Enterprise_Value_to_Sales', 'Price_To_Sales_Per_Share',
                 'Company_Shares', 'MCap_group', 'Enterprise_Value', 'gicsSectorName',
                 'gicsIndustryGroupName', 'gicsIndustryName', 'gicsSubIndustryName',
                 'Free_Cash_Flow_net_of_Dividends_per_Share',
                 'C_Price_to_FCF_net_of_Dividends_per_Share',
                 'Enterprise_Value_to_EBITDA', 'Enterprise_Value_to_Operating_Cash_Flow',
                 'Price_to_Cash_Flow_per_Share', 'P_to_E']

gics_names = ['gicsSectorName', 'gicsIndustryGroupName', 'gicsIndustryName', 'gicsSubIndustryName']

useless_columns = ['MCap_group']

util_columns = ['Date_round', 'RanksID', 'Perf_1Y']
cat_columns = gics_names + ['Country_of_Headquarters']
reg_columns = [el for el in valid_columns if el not in useless_columns + cat_columns + util_columns]
##########

data = pd.read_csv(join(data_path, 'clear_data.csv'), parse_dates=['Date_round'])

mask = get_inliers_mask_target(data)

# new_data = data[mask]
new_data = data.copy()
print('Train set:', new_data.shape[0] / data.shape[0])

not_outlier_ratios = []
for fn in reg_columns:
    arr = new_data[fn].to_numpy()
    f_arr = arr[get_inliers_mask(arr, lq, rq)]
    not_outlier_ratio = f_arr.shape[0] / arr.shape[0]
    not_outlier_ratios.append((not_outlier_ratio, fn))

filter_names = [fn for ratio, fn in not_outlier_ratios if ratio > 0.95]
mask = np.ones(new_data.shape[0], dtype=bool)
for fn in filter_names:
    arr = new_data[fn].to_numpy()
    mask &= get_inliers_mask(arr, lq, rq)

print('Train set:', new_data[mask].shape[0] / new_data.shape[0])
new_data = new_data[mask]
print('Train set:', new_data.shape[0] / data.shape[0])
new_data.to_csv(join(data_path, 'clear_outliers.csv'), index=False)
##################
print()
test = pd.read_csv(join(data_path, 'clear_test_data.csv'), parse_dates=['Date_round'])
mask = get_inliers_mask_target(test)
# new_test = test[mask]
new_test = test.copy()
print('Test set:', new_test.shape[0] / test.shape[0])
mask = np.ones(new_test.shape[0], dtype=bool)
for fn in reg_columns:
    arr = new_data[fn].to_numpy()
    lb, ub = get_inliers_borders(arr, lq, rq)
    mask &= (lb < new_test[fn]) & (new_test[fn] < ub)

print('Train set:', new_test[mask].shape[0] / new_test.shape[0])
new_test = new_test[mask]
print('Test set:', new_test.shape[0] / test.shape[0])
new_test.to_csv(join(data_path, 'clear_test_outliers.csv'), index=False)
