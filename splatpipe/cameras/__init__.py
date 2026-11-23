"""The Cameras_Step, which solves the cameras for the Picks that the Frames_Step wrote to
work/frames/: where each Pick was taken from, and the one lens that every Pick shares.

The step runs COLMAP 4.2.1 on the CPU, as the separate program colmap: feature_extractor on the
Grey_Copies, sequential_matcher twice, view_graph_calibrator, global_mapper, the incremental
mapper once when the global mapper gives up, and image_undistorter and
image_undistorter_standalone to one PINHOLE camera. Then it levels and centres the scene with one
rigid move, writes the levelled cameras and points to its Work_Folder, work/cameras/, for the
Training_Step, and copies the levelled cameras to out/cameras.json for the Viewer. Its settings
are in [cameras] in config.toml, and settings.py reads and checks them.

Every file in this package imports only the standard library, numpy, cv2 and splatpipe.
"""
