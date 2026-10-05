<!-- The title is a Conventional Commit (`feat(metal): few-query verify attention`):
it becomes the squash commit's subject and the release notes' line. This
description becomes the commit's body: it is the record of the work. -->

## What and why

<!-- What changed, for whom, and why now. Link the documents this changed. -->

## Evidence

<!-- What shows it works: the checks and gates run locally (the GPU and model
gates do not run in CI) with their result lines, measured numbers with their
record files under docs/benchmarks/, screenshots for the site or the agent. -->

## Remaining

<!-- What this leaves open, measured limits, follow-ups. "None" is an answer. -->

## Checklist

- [ ] `make verify-auto` passes from the branch's base, and the tiers it names ran or are exempt (say which and why)
- [ ] Documents that describe the change are updated; `make docs-check` passes
- [ ] The section of `TODO.md` this work came from is removed or updated
