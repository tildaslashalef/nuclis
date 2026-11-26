<!-- kind: record -->
# Prompt profiles: the record

The dated record behind [the reference](../../engine/prompt-profile.md).

## How the fixtures were captured (2026-09-07 to 2026-09-19)

The captures were taken by `scripts/tokenizer-fixtures.py` and
`scripts/profile-tools-fixtures.py` (retired; at tag `v0.11.0`) against
llama.cpp's server, which was the reference while the first families were
brought up ([models § 4](../../models/README.md#4-validate-against-nuclis-itself));
a re-pinned file's fixtures are made by nuclis
([models § Which executor makes which fixture](../../models/README.md#which-executor-makes-which-fixture)).

The script verified the model's full checksum, size, and template digest
(the server reports the template without the file's trailing newline;
both forms were accepted) and the reference build id. It passed the
thinking settings per request as the template's own keyword arguments
(`enable_thinking` and, for Qwen3.8, `reasoning_effort` and
`preserve_thinking`) and used `/apply-template`, `/tokenize`, and
`/detokenize`, with synthetic content only and no generation.

Captures: Qwen3.8 on 2026-09-07 (re-captured 2026-09-12 with the seven
conversations, 2026-09-19 with `high`), Gemma 4 on 2026-09-12, Muse
Glimmer on 2026-09-19, all with llama.cpp
`7620399f58aebfd2196b74021f9581bcf7218cb9` on the pinned files
([catalogue.md](catalogue.md#pinned-commits-and-digests-2026-09-11)).
The templates are adapted, not copied; their origins and licenses are in
[THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md).

## Template revisions checked (2026-09-17, 2026-09-18)

Two revisions were checked with `scripts/profile-alias-check.py` (retired;
at tag `v0.11.0`), which replayed every pinned case through llama.cpp's
server holding the file. On 2026-09-17, for the 17,530-byte Gemma 4 revision that community Gemma 4 12B
finetunes ship
(`dc311bb0…`): **18 of 38 cases differ**, all in history rendering — a
thought on an earlier tool-call step is dropped (the pinned template keeps
it), two consecutive assistant messages become two turns, and a tool-result
group before a user turn is left without its `<turn|>` — so it is not an
alias. On 2026-09-18, for Bonsai 2 27B's template (`c3cf9e34…`, 8,952 bytes: the
upstream Qwen3.8 template without Unsloth's fixes) on the PrismML fork's
server (`--reference-revision 5d80cff0…`; the script reported a template's
`raise_exception` as a mismatch rather than crashing): **4 of 68 cases
differ**, the `merged_system_*` histories, which that template refuses
(`System message must be at the beginning.`) where the pinned one merges
them; every other prompt and token stream is byte-identical
([bonsai.md § Chat template](../../models/bonsai.md#chat-template)). Not an alias
either; the Bonsai entry's profile was chosen from that evidence.

## Inkling 2's tokenizer and template (2026-11-26)

On `Inkling-2-Mini-UD-Q4_K_XL.gguf` at the pinned revision:

- `nuclis tokenize --raw --json` on 308 strings (scripts, cases, marks,
  contractions, digits, whitespace, emoji, code, two chat shapes) gave the
  same ids as the format's reference tokenizer on every one. That tokenizer
  splits with the folded classes `llama4` uses, and no corpus string tells
  the two apart: BPE seldom merges across the seams where they differ. Of
  thirteen words built on such a seam (`ÉCOLE`, `ÜBER`, `ÉTATS-UNIS`, …)
  one, `ÇA VA`, gives other ids; the fixture keeps it.
- The 16 prompts of the hand-written `inkling2-*.json` fixtures matched the
  format's reference rendering of the template byte for byte, images aside:
  there the reference puts its media marker where the template writes
  `<|content_image|><|unused_200054|>`. The fixture's ids matched its
  tokenizer except on `ÇA VA`.
