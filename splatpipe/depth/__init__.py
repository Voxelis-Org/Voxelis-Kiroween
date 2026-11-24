"""The Depth_Step, which makes depth priors for the Train_Step from the camera solve's frames.

The step runs MoGe-2 on each Frame of the Camera_Solve_Output in work/cameras/, fits each Frame's
inverse depth to the camera solve's own points, and writes the Frames that fit as Prior_Files,
then the Prior_Index, in work/depth/. Its settings are in [depth] in config.toml, and
settings.py reads and checks them.

The step's own code, `work`, `run` and the CLI, isn't in the package yet, so the package can't
be run as a step.
"""
