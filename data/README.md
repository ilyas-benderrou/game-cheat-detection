# Data folder

This folder is empty on GitHub on purpose (the data is not redistributed here).

1. Download the Kaggle dataset **"Online Gaming Behavior & Cheating Metrics 2026"**
   by masa.nakashima — https://www.kaggle.com/datasets/masanakashima/online-gaming-behavior-2026
   (licence: CC0 Public Domain). The file in that dataset is byte-for-byte identical to the one used
   here (SHA-256 `97ffc32e…ef0758`, checked on 2026-11-12). Without a Kaggle account you can try:

   ```bash
   curl -L -o data/oghb.zip https://www.kaggle.com/api/v1/datasets/download/masanakashima/online-gaming-behavior-2026
   ```
   then unzip it in this folder.
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
