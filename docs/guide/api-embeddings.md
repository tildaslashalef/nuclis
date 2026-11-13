<!-- kind: guide -->
# Embeddings

The embeddings route of [the nuclis API](api.md), which covers running
the server, its memory budget, its limits, and its errors.

## `POST /v1/embeddings`

OpenAI's embeddings call for the embedding models `nuclis embed` runs
(`embeddinggemma-2`, [embeddinggemma.md](../models/embeddinggemma.md)):
one unit vector per input, in input order. OpenAI's SDKs work unchanged:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="none")
r = client.embeddings.create(model="embeddinggemma-2", input=["a cat", "a kitten"])
q = client.embeddings.create(model="embeddinggemma-2", input="how do auroras form?",
                             extra_body={"task": "search_query"})
```

```sh
curl -s localhost:9000/v1/embeddings -d '{"input": "how do auroras form?", "task": "search_query"}'
```

## The embeddings request

| Field | Meaning |
| --- | --- |
| `model` | a registry entry of kind `embedding` or an embedding catalogue name; default `embed.model` (`embeddinggemma-2`). Never a path |
| `input` | a string (one input), or a list whose items are strings or lists of content parts, at most 2,048. A part is `{"type": "text", "text": "…"}`, `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}` (a base64 data URL or bare base64; PNG, JPEG, HEIC, WebP, TIFF, GIF, BMP), or `{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}` (`wav`, `mp3`, `aiff`, `flac`, or `m4a`; at most 30 s). Each item is one input and gives one vector; its parts are read in order, in one pass. The body limit (32 MiB) bounds the media |
| `dimensions` | `768` (default), `512`, `256`, or `128`: the vector's leading values, renormalized (the widths the model is trained for) |
| `encoding_format` | `float` (default) or `base64`: the vector's little-endian f32 bytes, which OpenAI's Python SDK asks for and decodes |
| `user` | accepted and ignored |
| `task` | nuclis's: the use the vectors are for. Text is embedded exactly as given unless a task is named; `search_query`, `question_answering`, `fact_checking`, `code_retrieval`, `classification`, `clustering`, and `similarity` open each input with the model's `task: … \| query: ` prefix, before any image or clip, `document` with `title: {title or none} \| text: ` ([embeddinggemma.md § Tokenizer and task prefixes](../models/embeddinggemma.md#tokenizer-and-task-prefixes)) |
| `title` | nuclis's: a document's title, with `"task": "document"` only |
| `truncate` | nuclis's: `true` cuts an input over 8,192 tokens at its end instead of refusing it, and a clip over 30 s to its first 30 s (Google's processor's cut); the response lists the inputs cut. An image or clip the cut would split is dropped whole |
| `image_tokens` | nuclis's: soft tokens per image, `70`, `140`, `280` (default), `560`, or `1120`, Google's processor's budgets. More is finer and slower; it changes the vector, and the response records it |

For retrieval, embed the corpus with `"task": "document"` and the queries
with `"task": "search_query"`: the two are trained to meet.

### What an embedding request may not carry

- **Token arrays** (`[1, 2, 3]` or `[[1, 2], [3]]`): a client that sends
  them tokenized with another model's vocabulary, so the vectors would be
  noise. LangChain's `OpenAIEmbeddings` does this by default; set
  `check_embedding_ctx_length=False` and it sends text.
- **An image by URL** (`https://…`): the server fetches nothing; send a
  data URL. An image that does not decode is `invalid_image`, a clip
  `invalid_audio`, each naming the input. An audio format other than the
  five is `unsupported_feature`.
- **A clip over 30 s** without `truncate` (`input_too_long`).
- **Images and audio without the projector**: `nuclis model pull
  embeddinggemma-2 --with mmproj` fetches it (`unsupported_feature` until
  then).
- **Other widths**, and an input over 8,192 tokens without `truncate`
  (`input_too_long`, with the input's index and token count).

## The embeddings response

OpenAI's shape on one line, then the `nuclis` object:

```json
{"object": "list",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.014093, -0.0013, …]}],
 "model": "embeddinggemma-2",
 "usage": {"prompt_tokens": 25, "total_tokens": 25},
 "nuclis": {"space": "embeddinggemma-2@6f1bd4ac6c5d/768", "dimensions": 768, "task": "search_query",
            "image_tokens": null, "tokens": [25], "truncated": [],
            "timings_ms": {"load": 0.0, "tokenize": 0.3, "embed": 58.9}}}
```

- A `float` value is the shortest decimal that reads back to the same
  f32; `base64` carries those bits exactly. Read as f32, the two are
  identical.
- `usage` counts every token the model read, the `<bos>` and `<eos>` each
  input is framed with included.
- `space` names where the vectors live: the checkpoint, the first 12 hex
  digits of its file's SHA-256, and the width. Vectors compare only within
  one space; an index should store it beside them.
- `image_tokens` is the image budget when the request held an image, else
  null. `tokens` is each input's token count (an image's soft tokens and
  its two markers included), `truncated` the indexes of the inputs that
  were cut, `timings_ms` the request's open (when it opened the model),
  tokenizing and image encoding, and the passes it was in.

## Passes and memory

Embedding requests share GPU passes of at most 2,048 rows (an input's
tokens, aligned to 8). When the GPU frees, the server takes the oldest
waiting request's model and fills a pass with the inputs of every request
waiting for that model, in arrival order; a request whose inputs do not
fit continues in the next pass, ahead of later ones, and an input longer
than a pass runs alone. A pass runs between a chat generation's steps
(about 0.2 s at 2,048 rows), so a generation and an indexing job share
the GPU. An input's vector does not depend on what shares its pass:
batching changes timing, never vectors (`scripts/app/api-check.py --only
embeddings` checks it bit for bit).

One embedding model is open at a time; it opens on its first request
(0.7 s) and counts its file against the [memory](api.md#memory) budget (310 MB
for the Q8_0 file). Its projector is opened with it when pulled; the
vision encoder is built on the first image (918 MB on Metal) and the audio
encoder on the first clip (658 MB), and the budget then adds each to the
model's share. Media are encoded one at a time on the GPU before their
pass: about 460 ms an image at 280 tokens, 130 ms for 4.8 s of speech.

