# Seed sweep

5 seeds per cell, 150 epoch budget, identical splits within a group.
A directional edge that only shows up for one seed is noise, not skill.

| group | model | runs | direction acc (mean +/- std) | MAE % (mean +/- std) | RMSE % |
|---|---|---|---|---|---|
| cross_sector | transformer | 5 | 0.531 +/- 0.019 | 2.649 +/- 0.121 | 3.822 |
| cross_sector | lstm | 5 | 0.529 +/- 0.005 | 2.547 +/- 0.042 | 3.727 |
| same_sector | transformer | 5 | 0.566 +/- 0.027 | 2.048 +/- 0.034 | 2.826 |
| same_sector | lstm | 5 | 0.477 +/- 0.014 | 2.149 +/- 0.052 | 2.878 |
