# TODO — active plan

This file is the queue: the units agreed and not yet merged, each with its
design, and a *Where we are* note. A unit is one session, one branch, and
one pull request, whose description is the record of the work; when it
merges, its section is deleted here, and when the last one goes, this file
is emptied back to this header and its *Deferred* list. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.

## Where we are

No unit is in progress. **EmbeddingGemma 2** (text, image and audio
embeddings; branch `embedding-gemma-2`) is complete and waits for review
in its pull request, which holds the record. **When it merges, release
0.7.0** (user, 2026-10-09): the release procedure in AGENTS.md
§ Versioning and releases, with `verify-long` and `verify-release` run
before the version pull request (`verify-cpu` ran on the branch). Nothing
else is queued: after the release, ask what to work on next.

Decided at its close (2026-10-09), and recorded in spec §10: nuclis stays
an engine. A RAG application calls `POST /v1/embeddings` and owns walking
files, chunking, storage and search; there is no index theme inside
nuclis.

## Deferred

Agreed and not scheduled; when one is picked up, it becomes a unit above.

- **Video input for EmbeddingGemma 2** (about one session). Frames at
  1 fps (at most 32) through the image encoder at 140 soft tokens each,
  an AVFoundation decoder beside the image and audio bridges, oracle
  fixtures from a short generated video, gates, `--video`, and a video
  part in the API. Google's processor facts: `docs/models/embeddinggemma.md`
  § The mmproj (`Video`).
