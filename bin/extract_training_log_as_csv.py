# heavily adapted https://stackoverflow.com/questions/71239557/export-tensorboard-with-pytorch-data-into-csv-with-python

# NOTE: best to move this file to the training log results folder


import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
from numpy import genfromtxt


# Extraction function
def tflog2pandas(path):
    if True:
        path = path
        runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
        try:
            event_acc = EventAccumulator(path)
            event_acc.Reload()
            tags = event_acc.Tags()["scalars"]
            for tag in tags:
                if tag in ["epoch","hp_metric","train_loss_step"]:
                    print(f"Skipping metric {tag}.")
                    continue
                print(f"writing metric {tag}.")
                event_list = event_acc.Scalars(tag)
                values = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x: x.step, event_list))
                r = {"metric": [tag] * len(step), "value": values, "step": step}
                r = pd.DataFrame(r)
                runlog_data = pd.concat([runlog_data, r])
            # Dirty catch of DataLossError
        except Exception:
            print("Event file possibly corrupt: {}".format(path))
            traceback.print_exc()


    # additionally add the class metrics for each epoch, from the class folder:
    # obviously really naive way of parsing the data...
    path+="class-metrics/"
    epoch_nrs = []
    metrics = []
    for p in Path(path).glob("*.csv"):
        filename = p.name.split(".")[0]
        l = filename.split("_")
        # metric, train/val, epochXXX
        epoch_str = l[2].rstrip('0123456789')
        epoch_nr = l[2][len(epoch_str):]
        epoch_nrs.append(int(epoch_nr))
        metrics.append(l[0])
    metrics = set(metrics) # remove duplicates

    print(f"additionally writing class metrics: {metrics}")
    print(f"Found data for epochs {min(epoch_nrs)} - {max(epoch_nrs)}.")

    epochs = max(epoch_nrs) - min(epoch_nrs) + 1
    step_size_per_epoch = runlog_data[runlog_data.metric == "Train/jaccard"].iloc[0]["step"] # really ugly way to get the step size
    for s in ["train","val"]:
        for m in (metrics):
            for e in range(epochs):
                vs = list(genfromtxt(path+m+"_"+s+"_epoch"+str(e)+".csv"))
                for idx,v in enumerate(vs):
                    if s == "train":
                        metric_name = "Train/"+m
                    elif s == "val":
                        metric_name = "Validation/"+m
                    step = e * (step_size_per_epoch+1) + step_size_per_epoch
                    r = {"metric": [(metric_name+"-"+str(idx))], "value": v, "step": step}
                    r = pd.DataFrame(r)
                    runlog_data = pd.concat([runlog_data, r])

    return runlog_data

pivot_table = True
dataset_results_foldername = "HCV2"
version_names = ["version_2", "version_3"]
for version_name in version_names:
    path="./"+dataset_results_foldername+"/lightning_logs/"+version_name+"/" #folderpath
    df=tflog2pandas(path)
    #df=df[(df.metric != 'params/lr')&(df.metric != 'params/mm')&(df.metric != 'train/loss')] #delete the mentioned rows
    if pivot_table:
        print("pivoting table")
        df = pd.pivot_table(df, values="value", columns="metric", index="step", sort=False)
    df.to_csv("results-"+dataset_results_foldername+"-"+version_name+".csv")

