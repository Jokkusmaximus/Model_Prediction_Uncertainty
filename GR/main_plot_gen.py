"""
Created on 16.05.24
by: jokkus
"""
from supplementary.tools import visualize_per_rollout, visualize_action_distribution
from datetime import datetime
from time import time

import seaborn as sns

# ** For timing
start_time = datetime.now()
start_time2 = time()
# -------------------------------------------


# savepaths = ["logs/special_cases/const_logstd_24_1720976154.5299006_0.1/",
#              "logs/special_cases/const_logstd_24_1720976154.5299006_0.01/",
#              "logs/special_cases/const_logstd_24_1720976154.5299006_0.05/",
#              "logs/special_cases/const_logstd_24_1720976154.5299006_0.005/",
#              "logs/special_cases/const_logstd_24_1720976154.5299006_1/",
#              "logs/special_cases/const_logstd_24_1720976154.5299006_1.5/",
#              "logs/special_cases/const_logstd_24_1720976154.5299006_2/",]

savepaths = ["logs/special_cases/const_logstd_24_1720976154.5299006_0.1/"]

for savepath in savepaths:
    # visualize_per_rollout(savepath, 10, True)
    visualize_action_distribution(savepath)

# -------------------------------------------
# ** For timing
end_time = datetime.now()
end_time2 = time()
print(f"total time: {end_time - start_time}, {end_time2 - start_time2}")
