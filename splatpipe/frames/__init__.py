"""The Frames_Step, which picks the frames of the Job's video that the camera solve works from.

The step reads the Used_Part of the video through the ffmpeg command line, scores every frame,
cuts the still ends, drops the jolted frames and picks sharp frames spread over the camera's
movement. It also finds the pixels stuck to the camera, for the Mask. It writes the Picks, their
grey copies, the Mask and frames.json to its Work_Folder, work/frames/, where the
Camera_Solve_Step reads them. Its settings are in [frames] in config.toml, and settings.py reads
and checks them.

Every file in this package imports only the standard library, numpy, cv2 and splatpipe.
"""
