# Data folder

This folder is empty on GitHub on purpose (the data is not redistributed here).

1. Download the Kaggle dataset **"Online Gaming Behavior and Cheating Metrics"**
   (search this exact title on https://www.kaggle.com/datasets).
2. Put the file `online_gaming_behavior.csv` in this folder:

```
game-cheat-detection/
└── data/
    └── online_gaming_behavior.csv
```

Expected file: 5,000 rows, 12 columns
(`Player_ID, Game_Session_ID, Game_Mode, Region, Rank_Level, Session_Duration, APM,
KD_Ratio, Accuracy, Headshot_Rate, Items, Cheat_Likelihood`).

You can also give another path: `python src/anticheat_pipeline.py --data path/to/file.csv`.
