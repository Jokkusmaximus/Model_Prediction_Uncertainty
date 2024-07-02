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

visualize_per_rollout(2, True)


# -------------------------------------------
# ** For timing
end_time = datetime.now()
end_time2 = time()
print(f"total time: {end_time - start_time}, {end_time2 - start_time2}")
