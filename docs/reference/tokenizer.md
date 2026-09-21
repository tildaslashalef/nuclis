# Native tokenizer

The library exposes `vocabulary.load(allocator, document, directory, limits)` in
[vocabulary.zig](../../inference/src/tokenizer/vocabulary.zig). This implements
owned GPT-2-family vocabulary storage. The library also exposes `bpe.encodePiece`
and `bpe.decode` in [bpe.zig](../../inference/src/tokenizer/bpe.zig), composed
by `tokenizer.Encoder` with the splitter the vocabulary's `pre` label selects:
`qwen35` ([pre.zig](../../inference/src/tokenizer/pre.zig)), `llama4`
([gpt4o.zig](../../inference/src/tokenizer/gpt4o.zig), Muse Glimmer), and
Gemma 4's SPM-style `gemma4` ([gemma4.md § Tokenizer](gemma4.md#tokenizer)).
The explicit artifact check matches all [prompt/token fixtures](prompt-profile.md).

## Input and ownership

The caller supplies a parsed GGUF document and the same file's directory image,
starting at file offset zero and ending at `document.directory_bytes`. Keep the
file unchanged between parsing and reading that image. No tensor payload is
needed. The pure loader receives no Io; the caller owns file access and bounds
the directory read before allocation.

`gguf.ArrayReader` traverses string and int32 arrays with count, offset, input,
and per-string limits. Returned strings borrow the directory image. It does not
change inspection's lazy-array behavior or materialize unrelated metadata.

The vocabulary copies text and builds token-ID and merge-rank lookup tables in
an owned arena. Release it with `deinit`; its source document and directory may
be released immediately after loading. Token array positions remain IDs; merge
positions remain ranks, with lower ranks taking priority. `tokenId` takes the
stored encoded string. `mergeRank` takes the stored pair representation, two
encoded strings separated by one ASCII space. Neither method encodes raw text.

## Validation and limits

The loader requires `tokenizer.ggml.model=gpt2`, a nonempty UTF-8 `pre` label up
to 128 bytes, tokens and merges as string arrays, and matching int32 token types.
It retains the pre-tokenizer name without treating arbitrary names as supported
tokenization algorithms. It rejects empty or invalid UTF-8 tokens, unknown token
types, duplicate tokens/pairs, and merge strings without exactly two nonempty
space-separated parts. It does not yet validate the byte alphabet or that merge
parts correspond to available pieces.

BOS, EOS, and padding IDs are optional; present IDs must be unsigned and inside
the vocabulary. Token kinds preserve normal, unknown, control, user-defined,
unused, and byte categories, with one override the reference applies to every
vocabulary at load: the spellings `<|start|>`, `<|message|>`, `<|channel|>`,
and `<|constrain|>` become user-defined, so the encoder matches them in plain
text even with `parse_special=false` (Muse Glimmer carries the first two; the
Qwen and Gemma vocabularies carry none). Additional tokenizer options and
special-ID metadata are not interpreted yet; successful loading does not
establish full compatibility.

Defaults allow one million tokens and merges each, 64 MiB of copied token/merge
text, and the GGUF defaults of a 64 MiB directory and 1 MiB individual strings.
Entry arrays, lookup tables, pre-label storage, and arena capacity are additional
memory. The aggregate text limit is not a total allocator budget. Expected
failures return typed errors and release partially initialized state.

## Full text encoding

`tokenizer.Encoder.init(allocator, &vocabulary)` accepts `pre` = `qwen35`,
`llama4`, or `gemma4` (any other label is `UnsupportedPreTokenizer`) and
owns a sorted special-ID index. The borrowed immutable vocabulary must outlive
the encoder. Call `deinit` once. Initialization accepts at most 4,096 special
markers, each between 1 and 256 bytes.

`encoder.encode(allocator, text, parse_special, limits)` returns owned token IDs.
It validates UTF-8, partitions special markers, splits remaining text with the
selected splitter's rules, then runs BPE separately on each piece. It never adds
BOS/EOS; its policy corresponds to the fixtures' `add_special=false`.

Control and unknown markers are recognized only with `parse_special=true`.
User-defined markers are always recognized, matching the reference convention.
Marker types are processed by descending spelling length; equal-length types
use token-ID order. Previously claimed spans cannot be split by later markers.
This precedence differs from simply choosing the first marker found left to
right. The decoder's supported kinds remain separately documented below.

The allocation-free `qwen35` iterator uses ordered alternatives for contractions,
letter/combining-mark runs with an optional prefix, individual Unicode numbers,
punctuation with trailing CR/LF, and whitespace. It preserves all source bytes;
there is no Unicode normalization. Whitespace backtracking can leave one space
attached to a following word. Returned pieces borrow their input fragment.

The `llama4` iterator (the gpt-4o pattern the reference assigns to that
label) differs in four places: the contraction is a suffix of the word
(`don't` is one piece), letter runs are cut at a case seam, numbers go in
runs of up to three, and a punctuation run also absorbs trailing `/`. The
case seam is the reference's, not Unicode's: it folds every letter class to
one and rewrites uppercase as "letter that is not ASCII a–z" and lowercase as
"letter that is not ASCII A–Z", so `HelloWORLD` splits into `Hello` and
`WORLD` but `ABCÀÉ` stays whole and `ÀBC` splits into `À` and `BC`; combining
marks are not letters here (`e\u0301` is `e` then `\u0301`). The rules and
their evidence are in [muse-glimmer.md § Tokenizer](muse-glimmer.md#tokenizer).
No table change was needed: the seam only reads ASCII case.

Category data comes from pinned llama.cpp `unicode-data.cpp`, not the host's
Unicode library. The compact checked-in table contains 1,921 five-byte range
records (9,605 bytes) for letter, mark, number, and whitespace membership.
The table's provenance is recorded in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
To reproduce the table with the pinned reference checkout available:

```sh
python3 scripts/tokenizer-unicode.py
```

The script verifies source SHA-256
`95170cd1c105a5b41a1b2dce73b0fae8ce8011ef7897600828bb2babe8b26e5d`.
The generated table SHA-256 is
`462abaa4ae40898a30bdde54e46ea081a917c77ec7c786a0a7f7cdcc74c26ebd`.
Default tests require neither this script nor the external checkout.

Encoding defaults bound input to 1 MiB, output to 1,048,576 IDs, special-marker
search to a conservative 1 GiB comparison budget (each marker's pass over the
text is a vectorized search for its first byte, charged per 16-byte block,
plus one comparison per candidate: Muse Glimmer's 2,048 reserved markers
would otherwise exhaust the budget on 18 KB of text, as the acceptance run
of MODL-13 found), and aggregate BPE scanning to 64 MiB across the entire
call. Per-piece BPE limits also apply. Work limits may
reject expensive input below the byte/token limits. Temporary marker storage
is proportional to input bytes; output capacity may exceed its logical length.
All allocations are explicit, and partial failures release owned state. The
reference algorithms prioritize correctness over speed; no tokenizer throughput
claim has been measured.

## BPE and ID-to-byte decoding

`bpe.encodePiece(allocator, vocabulary, bytes, limits)` returns owned token IDs
for one piece already separated by the caller. It accepts arbitrary bytes,
including incomplete UTF-8. Each byte maps reversibly to one scalar in the GPT-2
byte alphabet: visible ranges 33–126, 161–172, and 174–255 keep their code points;
other bytes map in order to U+0100 onward. For example, space maps to `Ġ`.
This is an encoding of bytes, not Unicode normalization.

The reference repeatedly scans adjacent pieces, selects the lowest merge rank,
and resolves equal ranks at the leftmost occurrence. A linked list represented
by array indices keeps merged spans contiguous in one encoded buffer. It does
not use longest-token matching, and it does not shortcut merely because a whole
piece exists in the vocabulary. Final symbols must resolve to normal tokens;
missing tokens and unsupported kinds return errors instead of dropping bytes.

Defaults bound one piece to 64 KiB, output to 65,536 IDs, and pair scanning to
64 MiB of examined encoded pair bytes, including separators. The work budget
bounds repeated scanning even for adversarial input; exceeding it returns
`WorkLimitExceeded`. This is a simple correctness reference with potentially
quadratic scanning, not an optimized tokenizer. Scratch storage is proportional
to input length and released before returning. The caller frees returned IDs.
**Do not pass an entire prompt as one piece:** doing so permits merges across
boundaries required by the pre-tokenizer.

`bpe.decode(allocator, vocabulary, ids, special, limits)` returns owned raw
bytes. Normal tokens reverse the byte alphabet. Control and user-defined tokens
emit their literal spelling when `special=true` and are omitted otherwise.
Unknown, unused, and byte-category tokens are explicitly unsupported by this
initial GPT-2-family decoder. Invalid IDs and invalid encoded scalars fail.
Defaults allow one million input IDs and 4 MiB of output; allocation occurs only
after validating and counting the complete output.

Decoding neither strips BOS/EOS IDs nor applies text cleanup. A token can end
inside a UTF-8 character, so a future streaming renderer must buffer incomplete
characters across calls. Both APIs borrow an immutable vocabulary for the call
and retain no references after returning. Neither inserts special IDs.

The byte alphabet is the GPT-2 byte-level mapping the vocabulary is stored in
([notices](../../THIRD_PARTY_NOTICES.md)).
The scan-based merge core is an independent implementation, checked against the
[pinned reference](reference-baseline.md) and synthetic ordering examples.

## Validation evidence

Default tests use a tiny synthetic GGUF and cover ownership after source release,
IDs/ranks/types, missing or malformed metadata, duplicates, invalid strings and
special IDs, exact limits, truncation, and allocation failures. BPE tests cover
all 256 bytes, partial UTF-8, ranks versus longest match, leftmost ties, changing
neighbors, exact work limits, decoding policy, and allocation failures. Unicode/encoder tests add
combining marks, supplementary categories, contractions, whitespace backtracking,
marker precedence, cross-piece merge prevention, shared budgets, and cleanup. No model or GPU
is required. An explicit check uses an external pinned artifact, selecting
its expectations and fixture by the file's template digest (through the
profile when one exists; Muse Glimmer's digest is matched directly until
its profile lands):

```sh
zig build test-vocabulary --global-cache-dir .zig-cache/global -- \
  "$HOME/.nuclis/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"
```

The inference package also exposes `zig build test-vocabulary` independently
(the `*-vocabulary` gates of `gates.json` run it per pinned file). The helper
is separate from the shipped nuclis CLI; inspection and structural validation
do not automatically load the vocabulary.

The 2026-09-07 check loaded all 248,320 tokens and 247,587 merges and verified
the `qwen35` label, BOS/EOS/padding IDs, and six token IDs from the captured
fixtures. The extended check also matches six single-piece fixture cases (three
inputs captured in two modes), re-encodes all 116 distinct normal token pieces
observed in the fixtures, and decodes all 20 standalone and 20 prompt sequences
byte-for-byte. Full native encoding now also matches every captured standalone
and prompt sequence in Debug and ReleaseSafe. This does not verify the file
checksum or establish correctness for inputs outside the tested cases. Artifact identity remains pinned in the [spec](../spec.md).
