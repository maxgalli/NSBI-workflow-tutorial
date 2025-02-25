import pandas as pd
import torch
import numpy as np
import zuko
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        #flow = zuko.flows.NSF(features=len(columns), context=0, transforms=4)
        flow = zuko.flows.NSF(features=len(columns), context=0, transforms=3)
        flow.load_state_dict(torch.load(f"flows/flow_{process}.pt"))
        flow = flow.to(device)

        # compute prob in batches to avoid memory issues
        bs = 1024
        log_prob_list = []
        tot_batches = len(dataset) // bs
        for i in range(0, len(dataset), bs):
            print(f"Batch {i//bs}/{tot_batches}")
            data = torch.tensor(dataset[columns].values[i:i+bs]).float().to(device)
            log_prob = flow().log_prob(data)
            prob = torch.exp(log_prob).detach().cpu().numpy()
            log_prob_list.append(prob)
        prob = np.concatenate(log_prob_list)
        probs[process] = prob
    
    out_path = "flows/probs.pkl"
    with open(out_path, "wb") as f:
        pkl.dump(probs, f)
    print(f"Saved probs to {out_path}")
