"""splatvid: video -> 3D gaussian splat, implemented from scratch.

Stages:
  video.extract_frames  -> sharp, evenly spaced frames
  sfm.run_sfm           -> camera poses + sparse point cloud (SIFT + PnP + BA)
  train.train           -> optimized GaussianModel (differentiable rasterizer)
  export.save_ply/splat -> standard output formats + bundled web viewer
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
