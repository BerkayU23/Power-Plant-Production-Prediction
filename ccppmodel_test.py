import pickle
import pandas as pd

def main():

    with open('CCPP_Systems.pkl', 'rb') as f:
        saved_data = pickle.load(f)

    X_test_scaled = pd.read_csv("ccpptestdata.csv")
    model = saved_data['model']
    print(model.predict(X_test_scaled))

if __name__ == "__main__":
    main()
