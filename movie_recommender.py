from data_preparation import prepare_dataset
from gui import run_gui

if __name__ == "__main__":
    df, cv, vectors_sparse = prepare_dataset()
    run_gui(df, cv, vectors_sparse)