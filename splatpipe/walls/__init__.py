"""The Walls_Step: corrects each frame's depth prior with a perspective bas-relief change.

It runs between the Depth_Step and the Train_Step, reads the depth priors in work/depth/ and the
camera solve in work/cameras/, and writes its own priors to work/walls/ in the Prior_Format.
Task 9 of the walls spec writes the step here.
"""
