"""The Train_Step, which trains the Job's 3D Gaussian splat from the camera solve's files.

The step reads the Camera_Solve_Output in work/cameras/, trains the splat on the job machine's
GPU with gsplat 1.5.3, and writes it to out/train.ply, where the Clean_Step reads it. Its
settings are in [train] in config.toml, and settings.py reads and checks them.

Only the GPU_Module, gpu.py, may import torch, gsplat and fused-ssim. Every other module of the
package imports only the standard library, numpy, cv2 and splatpipe, so importing splatpipe.train
on a laptop imports no GPU package (Req 9.11, 9.12).
"""
