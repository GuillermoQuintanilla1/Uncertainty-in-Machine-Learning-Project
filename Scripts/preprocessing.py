import pandas as pd
import os
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms


def preprocess_data():

    df = pd.read_csv('../Data/archive/HAM10000_metadata.csv')
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    base_skin_dir = '../Data/archive'
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
    df['path'] = df['image_id'].map(imageid_path_dict.get)
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes.astype(int)

    # Check class imbalance:
    # print(df['cell_type'].value_counts())

    ## Made sure there are no missing or duplicate values.
    ## Decide which of the two methods to use for handling missing values in the 'age' column. The first method fills missing values with the mean age, while the second method drops rows with missing age values.
    # df['age'].fillna((df['age'].mean()), inplace=True)
    df = df.dropna(subset=['age'])

    ## Train-test split:
    train_df, test_df = train_test_split(
        df,
        test_size=0.10,
        stratify=df['cell_type_idx'],
        random_state=42
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1111,
        stratify=train_df['cell_type_idx'],
        random_state=42
    )

    return train_df, val_df, test_df