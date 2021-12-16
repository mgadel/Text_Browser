Note : this code is for training purpose only, therefore the current code is training model at each request. Savings models would greatly inprove the execution time. Also no code optimisation / execution time has been performed.

├── LICENSE
├── README.md          <- The top-level README for this project.
├── config             <- configuration file at format JSON
├── data
│   ├── external       <- Data from third party sources.
│   ├── processed      <- Intermediate processed data.
│   └── raw            <- The original data.
│
├── dataloader         <- loading,, cleaning, transforming data
├── models             <- models (untrainned)
├── notebook           <- working notebook
├── test               <- code testing
├── utils              <- general data loading 
