# Dataset setup and attribution

## Source

This project uses the public **Credit Card Fraud Detection** dataset listed by [OpenML as dataset 1597](https://www.openml.org/d/1597). The downloadable parquet file is served from:

`https://data.openml.org/datasets/0000/1597/dataset_1597.pq`

The dataset page documents the data and its source context. It describes 284,807 European card transactions recorded over two days in September 2013, including 492 fraud examples. `V1`-`V28` are anonymized principal components, `Time` is elapsed time, `Amount` is the recorded amount, and `Class` is the binary outcome (`0` normal, `1` fraud).

Please retain attribution to OpenML and the dataset's original contributors when presenting or redistributing derived work. Check the [OpenML dataset page](https://www.openml.org/d/1597) for the current citation and terms; this project does not claim ownership of the source data.

## Local download behavior

The application downloads the parquet source file the first time `load_dataset()` runs and stores it here:

`data\creditcard.parquet`

The downloaded file is excluded by the root `.gitignore`; it is not part of this source project. An internet connection is needed for the initial download. If the file is incomplete or unreadable, remove only `data\creditcard.parquet` and restart the app so it can download a fresh copy.

## CSV scoring template

[`sample_transactions.csv`](./sample_transactions.csv) contains the required feature header only. Add real feature rows before upload. Do not copy the `Class` outcome into a prediction upload. Scoring requires exactly `Time`, `Amount`, and `V1` through `V28`, numeric finite values, and no negative `Time` or `Amount`.

The training dataset is provided with labels and is used only for the documented research experiments. Do not use the output as a real banking decision.
