# Seed sweep

5 seeds per cell, 150 epoch budget, identical splits within a group.
A directional edge that only shows up for one seed is noise, not skill.

| group | model | runs | direction acc (mean +/- std) | MAE % (mean +/- std) | RMSE % |
|---|---|---|---|---|---|
| cross_sector | transformer | 5 | 0.523 +/- 0.011 | 3.226 +/- 0.413 | 4.345 |
| cross_sector | lstm | 5 | 0.514 +/- 0.006 | 3.495 +/- 0.246 | 4.750 |
| same_sector | transformer | 5 | 0.487 +/- 0.057 | 2.716 +/- 0.396 | 3.499 |
| same_sector | lstm | 5 | 0.517 +/- 0.027 | 2.430 +/- 0.192 | 3.229 |
