import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        default_drop_columns = [
            'Id'
            , '3SsnPorch'
            , 'PoolArea'
            , 'BsmtFinSF2'
            , 'LowQualFinSF'
        ]

        if columns_to_drop is None:
            self.columns_to_drop = default_drop_columns
        else:
            self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1)

class ConvertCategoricalsToString(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_columns=None):
        default_categorical_columns = [
             'MSZoning',
             'MSSubClass',
             'Street',
             'Alley',
             'LotShape',
             'LandContour',
             'Utilities',
             'LotConfig',
             'LandSlope',
             'Neighborhood',
             'Condition1',
             'Condition2',
             'BldgType',
             'HouseStyle',
             'RoofStyle',
             'RoofMatl',
             'Exterior1st',
             'Exterior2nd',
             'MasVnrType',
             'ExterQual',
             'ExterCond',
             'Foundation',
             'BsmtQual',
             'BsmtCond',
             'BsmtExposure',
             'BsmtFinType1',
             'BsmtFinType2',
             'BsmtFullBath',
             'Heating',
             'HeatingQC',
             'CentralAir',
             'Electrical',
             'KitchenQual',
             'Functional',
             'FireplaceQu',
             'GarageType',
             'GarageFinish',
             'GarageQual',
             'GarageCond',
             'PavedDrive',
             'PoolQC',
             'Fence',
             'SaleType',
             'SaleCondition',
             'YrSold',
             'BsmtHalfBath',
             'MoSold',
             'KitchenAbvGr',
             'MiscFeature'
        ]

        if not categorical_columns:
            self.categorical_columns = default_categorical_columns
        else:
            self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_df = X.copy()

        for col in self.categorical_columns:
            new_df[col] = new_df[col].astype(str)

        return new_df

class ImputeMissings(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_impute=['LotFrontage', 'MasVnrArea', 'GarageYrBlt']):
        # self.columns_to_impute = columns_to_impute
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_df = X.copy()

        for col in new_df.columns:
            if new_df[col].dtype == np.dtype('O'):
                new_df[col] = new_df[col].fillna('Missing')
            else:
                new_df[col] = new_df[col].fillna(0)
        return new_df


class SubtractMinimum(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_subtract_from=['YearRemodAdd', 'YearBuilt']):
        self.columns_to_subtract_from = columns_to_subtract_from

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_df = X.copy()

        for col in self.columns_to_subtract_from:
            new_df[col] = new_df[col] - new_df[col].min()

        return new_df


class AddZeroIndicatorColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_add_indicators_for=None):
        default_columns = [
            'YearRemodAdd'
            , 'WoodDeckSF'
            , '2ndFlrSF'
            , 'MasVnrArea'
            , 'TotalBsmtSF'
            , 'BsmtUnfSF'
            , 'LotFrontage'
            , 'BsmtFinSF1'
        ]
        if columns_to_add_indicators_for is None:
            self.columns_to_add_indicators_for = default_columns
        else:
            self.columns_to_add_indicators_for = columns_to_add_indicators_for

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_df = X.copy()

        for col in self.columns_to_add_indicators_for:
            new_df['{}_zero'.format(col)] = 0
            new_df.ix[new_df[col] == 0, '{}_zero'.format(col)] = 1
            new_df['{}_zero_interaction'.format(col)] = new_df[col] * new_df['{}_zero'.format(col)]

        return new_df


class CorrectOverallCond(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_df = X.copy()
        new_df.ix[new_df['OverallCond'] > 5, 'OverallCond'] = 5

        return new_df


class StandardScaleContinuousFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, continuous_features=None):

        if continuous_features is None:
            self.continuous_features = [
                'GrLivArea'
                , 'TotalBsmtSF'
                , 'LotFrontage'
                , 'LotArea'
                , '1stFlrSF'
                , 'BsmtUnfSF'
                , 'MasVnrArea'
                , '2ndFlrSF'
                , 'YearRemodAdd'
                , 'BsmtFinSF1'
                , 'OverallQual'
                , 'OverallCond'
                , 'WoodDeckSF'
                , 'YearRemodAdd'
                , 'YearBuilt'
                , 'GarageCars'
                , 'GarageYrBlt'
                , 'GarageArea'
                , 'HalfBath'
                , 'OpenPorchSF'
                , 'BedroomAbvGr'
                , 'FullBath'
                , 'TotRmsAbvGrd'
                , 'Fireplaces'
            ]

        else:
            self.continuous_features = continuous_features

        self.stdscl = StandardScaler()

    def fit(self, X, y=None):
        self.stdscl.fit(X[self.continuous_features])
        return self

    def transform(self, X):
        new_df = X.copy()
        std = StandardScaler()
        new_continuous_features = self.stdscl.transform(new_df[self.continuous_features])
        new_column_names = [col + '_std' for col in self.continuous_features]
        cont_feat_rescaled_df = pd.DataFrame(new_continuous_features, columns=new_column_names)
        new_df = new_df.drop(self.continuous_features, axis=1).reset_index(drop=True)
        new_df = pd.concat([new_df, cont_feat_rescaled_df], axis=1)

        return new_df


class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, drop_outliers=True, z_score_threshold=4, target_columns=None):
        self.drop_outliers = drop_outliers
        self.z_score_threshold = z_score_threshold

        if target_columns is None:
            self.target_columns = [
                '1stFlrSF'
                , 'LotArea'
                , 'LotFrontage'
                , 'TotalBsmtSF'
                , 'GrLivArea'
            ]
        else:
            self.target_columns = target_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_df = X.copy()

        if self.drop_outliers:
            for col in self.target_columns:
                new_df = new_df[new_df[col + '_std'].abs() <= self.z_score_threshold]


        return new_df


class OneHotEncodeCategoricals(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns=None):
        if categorical_columns is None:
            self.categorical_columns = [
                  'MSZoning'
                , 'MSSubClass'
                , 'Street'
                , 'Alley'
                , 'LotShape'
                , 'LandContour'
                , 'Utilities'
                , 'LotConfig'
                , 'LandSlope'
                , 'Neighborhood'
                , 'Condition1'
                , 'Condition2'
                , 'BldgType'
                , 'HouseStyle'
                , 'RoofStyle'
                , 'RoofMatl'
                , 'Exterior1st'
                , 'Exterior2nd'
                , 'MasVnrType'
                , 'ExterQual'
                , 'ExterCond'
                , 'Foundation'
                , 'BsmtQual'
                , 'BsmtCond'
                , 'BsmtExposure'
                , 'BsmtFinType1'
                , 'BsmtFinType2'
                , 'BsmtFullBath'
                , 'Heating'
                , 'HeatingQC'
                , 'CentralAir'
                , 'Electrical'
                , 'KitchenQual'
                , 'Functional'
                , 'FireplaceQu'
                , 'GarageType'
                , 'GarageFinish'
                , 'GarageQual'
                , 'GarageCond'
                , 'PavedDrive'
                , 'PoolQC'
                , 'Fence'
                , 'SaleType'
                , 'SaleCondition'
                , 'YrSold'
                , 'BsmtHalfBath'
                , 'MoSold'
                , 'KitchenAbvGr'
                , 'MiscFeature'
            ]

            self.expected_columns = [
                 'EnclosedPorch',
                 'ScreenPorch',
                 'SalePrice',
                 'YearRemodAdd_zero',
                 'WoodDeckSF_zero',
                 '2ndFlrSF_zero',
                 'MasVnrArea_zero',
                 'TotalBsmtSF_zero',
                 'BsmtUnfSF_zero',
                 'LotFrontage_zero',
                 'BsmtFinSF1_zero',
                 'GrLivArea_std',
                 'TotalBsmtSF_std',
                 'LotFrontage_std',
                 'LotArea_std',
                 '1stFlrSF_std',
                 'BsmtUnfSF_std',
                 'MasVnrArea_std',
                 '2ndFlrSF_std',
                 'YearRemodAdd_std',
                 'BsmtFinSF1_std',
                 'OverallQual_std',
                 'OverallCond_std',
                 'WoodDeckSF_std',
                 'YearRemodAdd_std',
                 'YearBuilt_std',
                 'GarageCars_std',
                 'GarageYrBlt_std',
                 'GarageArea_std',
                 'HalfBath_std',
                 'OpenPorchSF_std',
                 'BedroomAbvGr_std',
                 'FullBath_std',
                 'TotRmsAbvGrd_std',
                 'Fireplaces_std',
                 'MSZoning_C (all)',
                 'MSZoning_FV',
                 'MSZoning_RH',
                 'MSZoning_RL',
                 'MSZoning_RM',
                 'MSSubClass_120',
                 'MSSubClass_160',
                 'MSSubClass_180',
                 'MSSubClass_190',
                 'MSSubClass_20',
                 'MSSubClass_30',
                 'MSSubClass_40',
                 'MSSubClass_45',
                 'MSSubClass_50',
                 'MSSubClass_60',
                 'MSSubClass_70',
                 'MSSubClass_75',
                 'MSSubClass_80',
                 'MSSubClass_85',
                 'MSSubClass_90',
                 'Street_Grvl',
                 'Street_Pave',
                 'Alley_Grvl',
                 'Alley_Pave',
                 'Alley_nan',
                 'LotShape_IR1',
                 'LotShape_IR2',
                 'LotShape_IR3',
                 'LotShape_Reg',
                 'LandContour_Bnk',
                 'LandContour_HLS',
                 'LandContour_Low',
                 'LandContour_Lvl',
                 'Utilities_AllPub',
                 'Utilities_NoSeWa',
                 'LotConfig_Corner',
                 'LotConfig_CulDSac',
                 'LotConfig_FR2',
                 'LotConfig_FR3',
                 'LotConfig_Inside',
                 'LandSlope_Gtl',
                 'LandSlope_Mod',
                 'LandSlope_Sev',
                 'Neighborhood_Blmngtn',
                 'Neighborhood_Blueste',
                 'Neighborhood_BrDale',
                 'Neighborhood_BrkSide',
                 'Neighborhood_ClearCr',
                 'Neighborhood_CollgCr',
                 'Neighborhood_Crawfor',
                 'Neighborhood_Edwards',
                 'Neighborhood_Gilbert',
                 'Neighborhood_IDOTRR',
                 'Neighborhood_MeadowV',
                 'Neighborhood_Mitchel',
                 'Neighborhood_NAmes',
                 'Neighborhood_NPkVill',
                 'Neighborhood_NWAmes',
                 'Neighborhood_NoRidge',
                 'Neighborhood_NridgHt',
                 'Neighborhood_OldTown',
                 'Neighborhood_SWISU',
                 'Neighborhood_Sawyer',
                 'Neighborhood_SawyerW',
                 'Neighborhood_Somerst',
                 'Neighborhood_StoneBr',
                 'Neighborhood_Timber',
                 'Neighborhood_Veenker',
                 'Condition1_Artery',
                 'Condition1_Feedr',
                 'Condition1_Norm',
                 'Condition1_PosA',
                 'Condition1_PosN',
                 'Condition1_RRAe',
                 'Condition1_RRAn',
                 'Condition1_RRNe',
                 'Condition1_RRNn',
                 'Condition2_Artery',
                 'Condition2_Feedr',
                 'Condition2_Norm',
                 'Condition2_PosA',
                 'Condition2_PosN',
                 'Condition2_RRAe',
                 'Condition2_RRNn',
                 'BldgType_1Fam',
                 'BldgType_2fmCon',
                 'BldgType_Duplex',
                 'BldgType_Twnhs',
                 'BldgType_TwnhsE',
                 'HouseStyle_1.5Fin',
                 'HouseStyle_1.5Unf',
                 'HouseStyle_1Story',
                 'HouseStyle_2.5Fin',
                 'HouseStyle_2.5Unf',
                 'HouseStyle_2Story',
                 'HouseStyle_SFoyer',
                 'HouseStyle_SLvl',
                 'RoofStyle_Flat',
                 'RoofStyle_Gable',
                 'RoofStyle_Gambrel',
                 'RoofStyle_Hip',
                 'RoofStyle_Mansard',
                 'RoofStyle_Shed',
                 'RoofMatl_CompShg',
                 'RoofMatl_Membran',
                 'RoofMatl_Metal',
                 'RoofMatl_Roll',
                 'RoofMatl_Tar&Grv',
                 'RoofMatl_WdShake',
                 'RoofMatl_WdShngl',
                 'Exterior1st_AsbShng',
                 'Exterior1st_AsphShn',
                 'Exterior1st_BrkComm',
                 'Exterior1st_BrkFace',
                 'Exterior1st_CBlock',
                 'Exterior1st_CemntBd',
                 'Exterior1st_HdBoard',
                 'Exterior1st_ImStucc',
                 'Exterior1st_MetalSd',
                 'Exterior1st_Plywood',
                 'Exterior1st_Stucco',
                 'Exterior1st_VinylSd',
                 'Exterior1st_Wd Sdng',
                 'Exterior1st_WdShing',
                 'Exterior2nd_AsbShng',
                 'Exterior2nd_AsphShn',
                 'Exterior2nd_Brk Cmn',
                 'Exterior2nd_BrkFace',
                 'Exterior2nd_CBlock',
                 'Exterior2nd_CmentBd',
                 'Exterior2nd_HdBoard',
                 'Exterior2nd_ImStucc',
                 'Exterior2nd_MetalSd',
                 'Exterior2nd_Plywood',
                 'Exterior2nd_Stone',
                 'Exterior2nd_Stucco',
                 'Exterior2nd_VinylSd',
                 'Exterior2nd_Wd Sdng',
                 'Exterior2nd_Wd Shng',
                 'MasVnrType_BrkCmn',
                 'MasVnrType_BrkFace',
                 'MasVnrType_None',
                 'MasVnrType_Stone',
                 'MasVnrType_nan',
                 'ExterQual_Ex',
                 'ExterQual_Fa',
                 'ExterQual_Gd',
                 'ExterQual_TA',
                 'ExterCond_Ex',
                 'ExterCond_Fa',
                 'ExterCond_Gd',
                 'ExterCond_TA',
                 'Foundation_BrkTil',
                 'Foundation_CBlock',
                 'Foundation_PConc',
                 'Foundation_Slab',
                 'Foundation_Stone',
                 'Foundation_Wood',
                 'BsmtQual_Ex',
                 'BsmtQual_Fa',
                 'BsmtQual_Gd',
                 'BsmtQual_TA',
                 'BsmtQual_nan',
                 'BsmtCond_Fa',
                 'BsmtCond_Gd',
                 'BsmtCond_Po',
                 'BsmtCond_TA',
                 'BsmtCond_nan',
                 'BsmtExposure_Av',
                 'BsmtExposure_Gd',
                 'BsmtExposure_Mn',
                 'BsmtExposure_No',
                 'BsmtExposure_nan',
                 'BsmtFinType1_ALQ',
                 'BsmtFinType1_BLQ',
                 'BsmtFinType1_GLQ',
                 'BsmtFinType1_LwQ',
                 'BsmtFinType1_Rec',
                 'BsmtFinType1_Unf',
                 'BsmtFinType1_nan',
                 'BsmtFinType2_ALQ',
                 'BsmtFinType2_BLQ',
                 'BsmtFinType2_GLQ',
                 'BsmtFinType2_LwQ',
                 'BsmtFinType2_Rec',
                 'BsmtFinType2_Unf',
                 'BsmtFinType2_nan',
                 'BsmtFullBath_0',
                 'BsmtFullBath_1',
                 'BsmtFullBath_2',
                 'Heating_Floor',
                 'Heating_GasA',
                 'Heating_GasW',
                 'Heating_Grav',
                 'Heating_OthW',
                 'Heating_Wall',
                 'HeatingQC_Ex',
                 'HeatingQC_Fa',
                 'HeatingQC_Gd',
                 'HeatingQC_Po',
                 'HeatingQC_TA',
                 'CentralAir_N',
                 'CentralAir_Y',
                 'Electrical_FuseA',
                 'Electrical_FuseF',
                 'Electrical_FuseP',
                 'Electrical_Mix',
                 'Electrical_SBrkr',
                 'Electrical_nan',
                 'KitchenQual_Ex',
                 'KitchenQual_Fa',
                 'KitchenQual_Gd',
                 'KitchenQual_TA',
                 'Functional_Maj1',
                 'Functional_Maj2',
                 'Functional_Min1',
                 'Functional_Min2',
                 'Functional_Mod',
                 'Functional_Sev',
                 'Functional_Typ',
                 'FireplaceQu_Ex',
                 'FireplaceQu_Fa',
                 'FireplaceQu_Gd',
                 'FireplaceQu_Po',
                 'FireplaceQu_TA',
                 'FireplaceQu_nan',
                 'GarageType_2Types',
                 'GarageType_Attchd',
                 'GarageType_Basment',
                 'GarageType_BuiltIn',
                 'GarageType_CarPort',
                 'GarageType_Detchd',
                 'GarageType_nan',
                 'GarageFinish_Fin',
                 'GarageFinish_RFn',
                 'GarageFinish_Unf',
                 'GarageFinish_nan',
                 'GarageQual_Ex',
                 'GarageQual_Fa',
                 'GarageQual_Gd',
                 'GarageQual_Po',
                 'GarageQual_TA',
                 'GarageQual_nan',
                 'GarageCond_Ex',
                 'GarageCond_Fa',
                 'GarageCond_Gd',
                 'GarageCond_Po',
                 'GarageCond_TA',
                 'GarageCond_nan',
                 'PavedDrive_N',
                 'PavedDrive_P',
                 'PavedDrive_Y',
                 'PoolQC_Fa',
                 'PoolQC_Gd',
                 'PoolQC_nan',
                 'Fence_GdPrv',
                 'Fence_GdWo',
                 'Fence_MnPrv',
                 'Fence_MnWw',
                 'Fence_nan',
                 'SaleType_COD',
                 'SaleType_CWD',
                 'SaleType_Con',
                 'SaleType_ConLD',
                 'SaleType_ConLI',
                 'SaleType_ConLw',
                 'SaleType_New',
                 'SaleType_Oth',
                 'SaleType_WD',
                 'SaleCondition_Abnorml',
                 'SaleCondition_AdjLand',
                 'SaleCondition_Alloca',
                 'SaleCondition_Family',
                 'SaleCondition_Normal',
                 'SaleCondition_Partial',
                 'YrSold_2006',
                 'YrSold_2007',
                 'YrSold_2008',
                 'YrSold_2009',
                 'YrSold_2010',
                 'BsmtHalfBath_0',
                 'BsmtHalfBath_1',
                 'BsmtHalfBath_2',
                 'MoSold_1',
                 'MoSold_10',
                 'MoSold_11',
                 'MoSold_12',
                 'MoSold_2',
                 'MoSold_3',
                 'MoSold_4',
                 'MoSold_5',
                 'MoSold_6',
                 'MoSold_7',
                 'MoSold_8',
                 'MoSold_9',
                 'KitchenAbvGr_1',
                 'KitchenAbvGr_2',
                 'KitchenAbvGr_3',
                 'MiscFeature_Gar2',
                 'MiscFeature_Othr',
                 'MiscFeature_Shed',
                 'MiscFeature_TenC',
                 'MiscFeature_nan',
                 'Condition2_RRAn',
                 'Exterior1st_Stone',
                 'Exterior2nd_Other',
                 'PoolQC_Ex',
                 'KitchenAbvGr_0'
                 ]
        else:
            self.categorical_columns = categorical_columns



    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_df = X.copy()

        # Ensure all of the categorical columns are strings
        for col in self.categorical_columns:
            new_df[col] = new_df[col].astype(str)

        # OHE the categorical features, then replace the old variables with the OHE features
        ohe_features = pd.get_dummies(new_df[self.categorical_columns])
        new_df = new_df.drop(self.categorical_columns, axis=1)
        new_df = pd.concat([new_df, ohe_features], axis=1)

        # Ensure all of the expected columns are in the DataFrame after creating the dummies
        for col in self.expected_columns:
            if col not in new_df.columns:
                new_df[col] = 0

        new_df = new_df[self.expected_columns]
        return new_df

class DataFrameCorrectOrder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_df = X.copy()
        new_df = new_df[self.expected_column_order]


class DataFrameExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values