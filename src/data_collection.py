import openml 
import pandas as pd
import os 


def data_collection():
    data = openml.datasets.get_dataset(46441)
    X, y, _, _ = data.get_data(target=data.default_target_attribute)
    df = pd.DataFrame(X)
    df['target'] = y
    return df
def save_data(df):
    data_path = os.path.join("data","raw")
    os.makedirs(data_path)
    output_path = os.path.join(data_path, "data.csv")
    df.to_csv(output_path, index =  False)

def main(): 
    data = data_collection()
    save_data(data)
if __name__ == "__main__":
    main()