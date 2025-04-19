import pandas as pd 
import logging
import pickle 
import os 

# logging configuration 
logging.basicConfig(level=logging.INFO) 

def save_from_csv(input_dir: str, output_dir: str): 
    if not os.path.exists(input_dir): 
        raise FileNotFoundError(f"{input_dir} not found.")

    df = pd.read_csv(input_dir, sep=r'\s+', header=None) 
    df.columns = ["B (Gauss)", "dI/dB (nA/G)"] 
    
    with open(output_dir, "wb") as f: 
        pickle.dump(df, f)

    logging.info(f"{output_dir} saved.")
    return 

def save_from_df(input_dir: str, output_dir: str, derivative=False): 
    if not os.path.exists(input_dir): 
        raise FileNotFoundError(f"{input_dir} not found.")

    df = pd.read_pickle(input_dir) 
    df.columns = ["B (Gauss)", "dI/dB (nA/G)"] if derivative else \
                 ["B (Gauss)", "I (nA)"]

    with open(output_dir, "wb") as f: 
        pickle.dump(df, f) 

    logging.info(f"{output_dir} saved.")
    print(df)
    return 


if __name__ == "__main__": 
    input_dir = "./raw/[EDMR]_DER_2G_3V.pkl" 
    output_dir= "./raw/[EDMR]_DER_2G_3V.pkl" 
    derivative = True 

    save_from_df(input_dir, output_dir, derivative)
