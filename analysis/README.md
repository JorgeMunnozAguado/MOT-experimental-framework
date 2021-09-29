
# Analysis of the outputs

Analysis of the results is perform in the notebook. Open the `analysis` environment and run the jupyter notebook with the following commands:

```
conda activate analysis
jupyter notebook
```

The analysis contain the following sections:

## Table analysis

Thanks to pandas library we can analize the tables of data in an easy way.

## Correlation matrix

Correlation matrices are used to calculate the association between detection and tracking metrics. To perform experiments with the given code follow the next steps:

1. Select a list of metrics to correlate them between them. `metrics = [...]`.
1. Calculate the correlation matrix with the *pandas* *DataFrame* and the selected metrics.
```python
result_m = aux.correlation_metrics(tb, metrics)
```
1. Plot the correlation matrix.
> You can specify the size of the plot `figsize=()`.
> The figure can be save in a folder if `file_name` is not *None*.
```python
aux.plot_matrix(result_m, metrics, file_name='correlation_matrix.png')
```

## Search 

To search results with similar scores use `search_correlation()`. This functions allow us, for example, to search experiments where *MOTA* is similar.
```python
search_correlation(tb, conditions, type_c)
```
Where:
- *tb* is the *pandas DataFrame*.
- `conditions` is a dictionary where to specify the search we want to perform. In the example bellow, we are searching cases where HOTA varies between two experiments in *0.05*:
```
{'HOTA': 0.05}
{'HOTA': 0.05, 'MOTA': 0.07}
```
- `type_c` decides how to merge conditions. See the examples below:
```
> search_correlation(tb, {'HOTA': 0.05, 'MOTA': 0.07, 'IDSW': 2}, ['and'])

      conditions HOTA and MOTA and IDSW.


> search_correlation(tb, {'HOTA': 0.05, 'MOTA': 0.07, 'IDSW': 2}, ['or'])

      conditions HOTA or MOTA or IDSW.


> search_correlation(tb, {'HOTA': 0.05, 'MOTA': 0.07, 'IDSW': 2}, ['and', 'or'])

      conditions HOTA and MOTA or IDSW.
```

> If you want to search in only one sequence use `search_seq()`.