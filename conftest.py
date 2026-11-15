"""Make the repo-relative 'out/' directory resolve to 'results/'.

The simulator and harness read pre-computed archives from 'out/' and
also write regenerated ones there. This shim creates 'out' as a link
(or a copy on platforms without symlink support) so a fresh clone
finds the tracked archives in results/ without any path edits.
"""
import os, shutil

root = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(root, "out")
results = os.path.join(root, "results")
if not os.path.exists(out):
    try:
        os.symlink("results", out)
    except (OSError, NotImplementedError):
        shutil.copytree(results, out)
