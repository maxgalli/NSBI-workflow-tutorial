import pandas as pd
import torch
import numpy as np
import zuko
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl


if __name__ == '__main__':
    processes = ["ttbar", "wjets", "single_top_t_chan"]
    #processes = ["wjets"]

    probs = {}
    for process in processes:
        print(f"Process: {process}")
        saved_data='./cached_data/'
        dataset = pd.read_hdf(saved_data + "dataset_preselected_ttbar.h5", "dataset")
        columns = ['log_lepton_pt', 'log_H_T', 'lepton_eta', 'lepton_phi']

        #dataset = dataset[dataset["type"] == process]
        # scale to (-5, 5)
        for column in columns:
            scaler = MinMaxScaler(feature_range=(-5, 5))
            dataset[column] = scaler.fit_transform(dataset[[column]]).flatten()

        # load the trained flow
        flow = zuko.flows.NSF(features=len(columns), context=0, transforms=4)
        flow.load_state_dict(torch.load(f"flows/flow_{process}.pt"))

        # compute prob
        log_prob = flow().log_prob(torch.tensor(dataset[columns].values).float())
        prob = torch.exp(log_prob).detach().numpy()
        print(prob)
        probs[process] = prob
    
    out_path = "flows/probs.pkl"
    with open(out_path, "wb") as f:
        pkl.dump(probs, f)
    print(f"Saved probs to {out_path}")