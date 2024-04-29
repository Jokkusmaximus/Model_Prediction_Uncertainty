# Model_Prediction_Uncertainty
Guided Research, investigating ways of extracting uncertainty from model prediction







# Hotfixes
Due to gymnasium=0.29 and Mujoco=3.14, there exists a breaking bug in mujoco_rendering.py 
Fix: line 593: "self.data.solver_iter" -> "self.data.solver_niter"
