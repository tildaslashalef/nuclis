<!-- Every pull request is merged by rebase, so each unit's record lives
in its commit body, never only here.

A milestone's pull request: title it for the milestone, list its units
below; the sections after "Units" then summarize the milestone.

A single unit outside a milestone: one commit; the title is that commit's
Conventional Commit subject (`feat(metal): few-query verify attention`) and
the sections after "Units" repeat its record. Delete the "Units" section. -->

## Units

<!-- One line per unit, in order: the commit's subject and its short hash. -->

## What and why

<!-- What changed, for whom, and why now. Link the documents this changed. -->

## Evidence

<!-- What shows it works: the checks and gates run locally (the GPU and model
gates do not run in CI) with their result lines, measured numbers with their
record files under docs/benchmarks/, screenshots for the site or the agent. -->

## Remaining

<!-- What this leaves open, measured limits, follow-ups. "None" is an answer. -->

## Checklist

- [ ] `make check` and `make build-linux` pass at the branch's tip (the GPU checks and the Linux build do not run per unit in CI)
- [ ] `make verify-auto` passes from the branch's base, and the tiers it names ran or are exempt (say which and why)
- [ ] Documents that describe the change are updated; `make docs-check` passes
