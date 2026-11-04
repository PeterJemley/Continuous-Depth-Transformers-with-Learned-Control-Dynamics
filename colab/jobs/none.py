"""The job NEXT_JOB.json names when no job is queued. It trains and measures nothing, and exits at once.
The runner still uploads its short log, so an accidental run leaves a record and costs a few minutes, not a GPU session."""
print('No job is queued, so nothing ran. Close this notebook: there is nothing to report.')
