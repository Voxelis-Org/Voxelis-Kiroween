"""The Clean_Step: it cleans the trained splat and makes the small copy for the web viewer.

It reads the splat the training step wrote, out/train.ply, and drops the Gaussians a viewer
shouldn't see: the broken ones, the ones too faint to add anything and the isolated floaters. It
writes the rest to out/clean.ply as a binary PLY, then runs splat-transform under Node to make
out/clean.sog for the web viewer. Every part of it is CPU work.

The file names and every threshold are settings in [clean] in config.toml. settings.py checks
them before the step opens, writes or removes anything.

From splatpipe, this package imports only splatpipe.step, splatpipe.errors and splatpipe.config.
Outside the standard library it imports only numpy and scipy.spatial, and nothing from cloud/,
boto3, torch or gsplat (Req 8.5).
"""
