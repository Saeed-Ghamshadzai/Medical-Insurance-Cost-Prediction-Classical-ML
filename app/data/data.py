import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class Preprocessor:
    """
    Class to preprocess input data for machine learning modeling.
    """

    def __init__(self):
        """
        Initialize Preprocessor object with required transformers.
        """
        self.mean_charges = {}
        self.label_encoder_age = LabelEncoder()
        self.label_encoder_bmi = LabelEncoder()
        self.age_bins = None
        self.bmi_bins = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.age_min_max = ()
        self.bmi_min_max = ()
        self.chl_min_max = ()

    def encode(self, data):
        """
        Encode categorical variables and generate new features for 'region'.

        Args:
            data (DataFrame): Input data containing 'sex', 'smoker', 'region' columns.

        Returns:
            DataFrame: Encoded data.
        """
        df = data.copy()

        for i in ['northeast', 'northwest', 'southeast', 'southwest']:
            region_encoder = lambda x: 1 if x==i else 0
            df[i] = df['region'].map(region_encoder)
            df.insert(6, i, df.pop(i))
        else:
            df.drop(columns=['region'], inplace=True)


        sex_encoder = lambda x: 1 if x=='male' else 0
        smoker_encoder = lambda x: 1 if x=='yes' else 0
        
        df['sex'] = df['sex'].map(sex_encoder)
        df['smoker'] = df['smoker'].map(smoker_encoder)

        return df

    def scaler(self, data):
        """
        Scale the features using MinMaxScaler.

        Args:
            data (DataFrame): Input data containing features to scale.

        Returns:
            DataFrame: Scaled data.
        """
        features, target = self._get_features_and_target(data)
        
        scaled_features = self.feature_scaler.transform(features)
        scaled_target = self.target_scaler.transform(target.values.reshape(-1, 1))

        data.loc[:, features.columns] = scaled_features
        data.loc[:, ['charges']] = scaled_target
        
        return data

    def fit(self, data):
        """
        Fit preprocessing transformers to the input data.

        Args:
            data (DataFrame): Input data to fit transformers on.
        """
        self.age_min_max = (data['age'].min(), data['age'].max())
        self.bmi_min_max = (data['bmi'].min(), data['bmi'].max())
        self.chl_min_max = (data['children'].min(), data['children'].max())
        
        data = self.encode(data)

        # Fit the transformations
        categorical_features = ['sex', 'children', 'smoker', 'northeast', 'northwest', 'southeast', 'southwest']
        
        for feature in categorical_features:
            self.mean_charges[feature] = data.groupby(feature)['charges'].mean()
            data[f'{feature}_mean_charges'] = data[feature].map(self.mean_charges[feature])

        self.age_bins = pd.cut(data['age'], bins=7).unique().categories
        self.bmi_bins = pd.cut(data['bmi'], bins=7).unique().categories
        data['age_bin'] = pd.cut(data['age'], bins=self.age_bins)
        data['bmi_bin'] = pd.cut(data['bmi'], bins=self.bmi_bins)

        label_encoder_age = self.label_encoder_age.fit_transform(data['age_bin'])
        label_encoder_bmi = self.label_encoder_bmi.fit_transform(data['bmi_bin'])

        data['age_bin_encoded'] = label_encoder_age
        data['bmi_bin_encoded'] = label_encoder_bmi

        # Drop the original binned columns
        data.drop(['age_bin', 'bmi_bin'], axis=1, inplace=True)
        
        # Fit the scalers
        features, target = self._get_features_and_target(data)
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.values.reshape(-1, 1))
    
    def transform(self, data, drop_target=True, scale=False):
        """
        Transform input data using the fitted transformers.

        Args:
            data (DataFrame): Input data to transform.
            drop_target (bool): Whether to drop the target variable 'charges'.
            scale (bool): Whether to scale the features.

        Returns:
            DataFrame: Transformed data.
        """
        data = self.encode(data)

        # Generate new features for mean 'charges' per category
        categorical_features = ['sex', 'children', 'smoker', 'northeast', 'northwest', 'southeast', 'southwest']
        
        for feature in categorical_features:
            data[f'{feature}_mean_charges'] = data[feature].map(self.mean_charges[feature])
        
        # Assign bin labels
        data['age_bin'] = pd.cut(data['age'], bins=self.age_bins)
        data['bmi_bin'] = pd.cut(data['bmi'], bins=self.bmi_bins)

        # Label encoding for the binned categories
        data['age_bin_encoded'] = self.label_encoder_age.transform(data['age_bin'])
        data['bmi_bin_encoded'] = self.label_encoder_bmi.transform(data['bmi_bin'])

        # Drop the original binned columns
        data.drop(['age_bin', 'bmi_bin'], axis=1, inplace=True)

        if scale:
            # Scale the features
            data = self.scaler(data)

        if drop_target:
            data.drop(columns=['charges'], inplace=True)
        
        return data
    
    def fit_transform(self, data):
        """
        Fit transformers to the input data and transform it.

        Args:
            data (DataFrame): Input data to fit and transform.

        Returns:
            DataFrame: Transformed data.
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform_target(self, scaled_target):
        """
        Inverse transform the scaled target variable to its original scale.

        Args:
            scaled_target (ndarray): Scaled target values.

        Returns:
            ndarray: Inverse transformed target values.
        """
        return self.target_scaler.inverse_transform(scaled_target)
    
    def inverse_transform_features(self, scaled_features):
        """
        Inverse transform the scaled features to their original scale.

        Args:
            scaled_features (ndarray): Scaled feature values.

        Returns:
            ndarray: Inverse transformed feature values.
        """
        return self.feature_scaler.inverse_transform(scaled_features)
    
    def _get_features_and_target(self, data):
        """
        Helper method to select feature columns (excluding target).

        Args:
            data (DataFrame): Input data.

        Returns:
            DataFrame: Features, DataFrame: Target.
        """
        return data.drop(columns=['charges']).astype(float), data['charges'].astype(float)

# Load environment variables
load_dotenv()
# # Get the model version from the environment
path_to_data = os.getenv('PATH_TO_DATASET')

csv = pd.read_csv(path_to_data)
df = pd.DataFrame(csv)

preprocessor = Preprocessor()
preprocessor.fit(df)