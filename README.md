## User Guide 
# run_pipeline.py
Main motion-synthesis pipeline.
Loads a base motion (base_motion.pt) and then splices in motion snippets from a dataset (ptFiles).
Uses nearest-neighbor pose matching to find similar poses, then blends snippets in with crossfades, root rotation control, translation smoothing, and safeguards against low-energy motions.
Produces an output .pt file (pipeline_output.pt) and an audit CSV log of which snippets were chosen.
# motion_simulator.py
Generates a procedural base motion for a character walking along user-defined waypoints.
Adds gait cycles (hip, knee, ankle, shoulder, elbow swings) to produce natural locomotion.
Outputs a .pt file with pose, translation, and betas tensors, which run_pipeline.py uses as the baseline.
For creating a controllable reference trajectory.
# motion_loader.py
Utility to load all poses from .pt files in a folder.
Extracts per-frame pose vectors into a list and a big numpy matrix, which are later used for nearest-neighbor search in pose_matcher.py.
Essentially converts a folder of motion capture files into a searchable database of poses.
# pose_matcher.py
Wraps scikit-learn’s NearestNeighbors for fast similarity search between poses.
Given a query pose, finds the closest matching poses in the database loaded by motion_loader.py.
This is the “search engine” that powers the snippet selection in run_pipeline.py.
# motion_blender.py
Provides simple blending utilities:
blend_poses: linear interpolation between two poses.
crossfade_tracks: crossfade two motion sequences over a fade window.
These functions are the building blocks for smooth transitions between snippets.
# snippet_utils.py
Collection of helper functions for rotation math and snippet handling.
Converts between axis-angle, rotation matrices, and quaternions; implements quaternion slerp.
load_snippet: extracts a contiguous snippet from a motion file (pose, translation, betas).
rebase_snippet_to_state_windowed: aligns a snippet to the current motion output (rotation + translation offset) to avoid jumps.
# inspect_pipeline.py 
Simply a log of the motion selections and switches. 
# visualise_smpl_motion.py
To visualise the motion.
