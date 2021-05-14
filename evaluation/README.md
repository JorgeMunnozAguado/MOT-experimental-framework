
# Evaluation framework

This tool was create for evaluating the detector and tracker outputs. In this readme you can find information of the structure of the tools and how to run them.

## Evaluate detectors and trackers

For evaluating the detectors run the following comand. It evaluate all the outputs over the possible datasets.
```
python evaluation/mAP/main.py
```


For evaluating the tracker run the following command. This command just evaluate a tracker over one dataset. To run over more trackers tune the flag `--TRACKERS_TO_EVAL`. To run over more datasets, tune the flag `--BENCHMARK`.
```
python evaluation/scripts/run_mot_challenge.py --BENCHMARK MOT17 --USE_PARALLEL True --NUM_PARALLEL_CORES 4 --TRACKERS_TO_EVAL sort
```


To recopile all the scores run the following command. It creates a table with all the tracking and detection metrics.

```
python evaluation/create_table.py
```

