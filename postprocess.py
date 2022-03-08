
from dedalus.tools import post

post.merge_process_files("snapshots", cleanup=True)

post.merge_process_files("analysis_tasks", cleanup=True)

import pathlib
set_paths = list(pathlib.Path("analysis_tasks").glob("analysis_tasks_s*.h5"))
print(set_paths)
post.merge_sets("analysis_tasks/analysis.h5", set_paths, cleanup=False)



