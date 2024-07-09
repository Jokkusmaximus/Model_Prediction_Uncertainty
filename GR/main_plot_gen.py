"""
Created on 16.05.24
by: jokkus
"""
from supplementary.tools import visualize_per_rollout
from datetime import datetime
from time import time

# ** For timing
start_time = datetime.now()
start_time2 = time()
# -------------------------------------------

savepaths = ["logs/seeds/rl_model_17_1719938269.4566817_0.1/",
             "logs/seeds/rl_model_17_1719938269.4566817_0.01/",
             "logs/seeds/rl_model_17_1719938269.4566817_0.05/",
             "logs/seeds/rl_model_17_1719938269.4566817_0.005/",
             "logs/seeds/rl_model_17_1719938269.4566817_1/",
             "logs/seeds/rl_model_17_1719938269.4566817_2/",]

for savepath in savepaths:
    visualize_per_rollout(savepath, 10, True)


# -------------------------------------------
# ** For timing
end_time = datetime.now()
end_time2 = time()
print(f"total time: {end_time - start_time}, {end_time2 - start_time2}")
