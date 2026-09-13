# Memory-lean MoE inference

Run a 35B mixture-of-experts model on Apple Silicon, on `mlx.core`.

A mixture-of-experts model activates a handful of experts per token but is
normally loaded whole: the validated checkpoint is 35B parameters across 256
experts per layer, of which 8 run. This engine keeps the always-on weights
resident and streams the routed experts off disk as routing picks them, so
the memory it holds is set by a cache size rather than by the model's size:
2.1 GiB at the default, for a checkpoint that is 19 GiB on disk. It runs on
an 8 GB M1, which is what the numbers below were measured on.

Everything else is ordinary: one OpenAI-compatible endpoint, so anything that
speaks that API — aider, and the rest of that ecosystem — points at it
without a plugin.

Requires Python 3.12 or newer, and Apple Silicon.

```
uv sync
```

## Download a model

Checkpoints go into the Hugging Face cache, laid out the hub's own way, so
`huggingface_hub` and every other tool reading that cache finds the same copy.

```
uv run python -m mlx_lean_moe.weights.download <repo id>
```

The download is resumable: re-running the same command continues where it
stopped rather than starting over.

`--connections` sets how many byte ranges of a file are in flight at once
(default 24) and `--revision` takes a branch or commit. The hub answers too
much concurrency with 429 rather than by slowing down, which the downloader
backs off from; if a download crawls, fewer connections may be faster.

The architecture this engine reads is 4-bit Qwen3.5-MoE (`qwen3_5_moe`); the
checkpoint it is validated against is
`froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit`, 19.0 GiB.

## Run it

```
uv run mlx-lean-moe
```

Serves `http://127.0.0.1:8080/v1` and prints the base URL; `--host` and
`--port` move it.

With one checkpoint downloaded that is the whole command — the server loads
it. `--model <repo id>` picks between several, and is required once there is
more than one: choosing for you would mean loading 19 GB of whichever came
first alphabetically.

Which model the *client* names is a separate thing, and this server ignores
it: it answers from the checkpoint it loaded and reports that name back, so
`--model` on the server side is what decides. There is one model per server.

Three flags are fixed when the model is built, so they live on the server
rather than in a request:

- `--max-context` is the attention context length (default 32768).
- `--expert-cache` is how many experts stay resident per layer, and is the
  memory dial (default: the top-k, so one slot per routed expert).
- `--think` lets the model emit a reasoning block before answering.

`python -m mlx_lean_moe` is the same entry point.

## What it costs

Measured on an 8 GB M1, macOS 26.5.2, mlx 0.32.2, against the checkpoint
above with its expert pack built: a 2106-token prompt, 60 tokens generated,
`--max-context 8192`, a fresh process per row. Peak memory below is from
that short context; see the note after the table for what a longer one adds.

| `--expert-cache` | peak memory | prefill | decode |
| --- | --- | --- | --- |
| 8 (default) | 2.10 GiB | 26 tok/s | 5.0 tok/s |
| 16 | 2.62 GiB | 27 tok/s | 5.0 tok/s |
| 32 | 3.65 GiB | 14–21 tok/s | 3.8–4.6 tok/s |

Peak memory is `mx.get_peak_memory()`, which reproduced within 0.01 GiB
across runs. Process RSS is not quoted because it did not: it ranged from
1.9 to 3.1 GiB for the same configuration.

Two things in that table are worth reading twice.

**The dial does not go "more is better".** At 32 experts per layer the engine
is both slower and erratic — four runs spread across 3.8–4.6 tok/s, where the
smaller caches held within 0.1 of each other. The erratic part is the clue,
and the likely explanation is that a larger cache leaves less room for the
page cache holding the weights it reads, so it buys hits and pays for them in
misses; that mechanism is a guess, but the slowdown is not. On a machine with
more memory the sweet spot will be elsewhere: measure rather than assume it is
the largest value that fits.

**Extra cache hits do not become speed here**, which is why the default is
one slot per routed expert rather than two. Over six prompts in different
languages and subjects, sixteen slots hit 44.8% against eight slots' 33.8% —
17% fewer reads reaching disk, a real improvement — and the two ran at 3.85
and 3.90 tok/s. The hit rate is deterministic and repeated exactly; the
throughput difference is nil.

The reading is that the reads an extra slot saves were being served from the
operating system's page cache anyway, so skipping them saves nothing while
the slot itself costs half a gigabyte. That will not hold on a machine whose
reads are genuinely slower, or with less room for the page cache — where
those 17% would start to count. `--expert-cache 16` is one flag away.

Context is cheap in this architecture and that is the point of it: only 10
of the 40 layers use softmax attention, with two key/value heads each, so a
token of history costs 20 KiB across the whole model — 0.62 GiB if the
default 32768 is filled to the brim, and nothing at all until it is, since
the buffers grow as the conversation does. The other 30 layers keep a
fixed-size recurrent state that does not grow with context.

The server is listening about two seconds after you start it, of which the
model is 0.7 s: nothing is loaded eagerly, the resident weights are read as
the first token needs them, and the routed experts never all arrive. The
first request of a fresh process is still slower than the rest, paying once
for Metal kernel compilation; the table above is measured after a warm-up
turn, as a served request would be.

## Point something at it

Any OpenAI client works — the API key is not checked, so send anything:

```
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}'
```

`GET /v1/models` lists the one loaded model. From the request body it honours
`messages`, `max_tokens`, `stream`, `stop`, `temperature`, `top_p`, `seed`,
`frequency_penalty`, `presence_penalty`, `logit_bias` and `n`. A value
outside its documented range is refused with a 400 naming the field, rather
than quietly clamped.

Decoding is greedy by default: `temperature: 0` is exactly `argmax`. The
penalties and `logit_bias` still apply there — they move which token *is* the
argmax — and both penalties count only generated tokens, not the prompt.

`n` runs that many completions one after another, since there is one model:
asking for three costs three answers' worth of time, and under greedy
decoding returns the same one three times. With a seed, each choice moves it
so the completions differ while the request as a whole still replays.

Not implemented: `tools`/`tool_choice`, `/v1/embeddings` and
`/v1/completions`. That decides which clients work — a coding agent that
drives its tools through plain text is fine, one that needs native function
calling is not.

One request runs at a time. The model lives on a single thread — MLX's
default stream is thread-local — and that thread is also what keeps two
answers from interleaving into one conversation state.

For example, with aider:

```
export OPENAI_API_KEY=anything
aider --openai-api-base http://127.0.0.1:8080/v1 --model openai/<repo id>
```

The server prints that line with its own address already filled in.

## Faster reads, at the cost of disk

Each expert is stored as nine separate byte ranges, so reading one costs nine
reads, and read count is what decode time follows. Repacking them into one
contiguous read each is worth about 6% of decode throughput.

```
uv run python -m mlx_lean_moe.weights.expert_pack <repo id>
```

It also takes a checkpoint directory, if the weights are somewhere the hub
cache is not.

This writes `experts.pack` beside the weights, roughly doubling what the
checkpoint occupies (16.9 GiB for the default one, on top of its 19 GiB). The
engine picks it up automatically when it is there. It is a derived file:
deleting it changes nothing but speed.

## Delete a model

```
uv run hf cache ls
uv run hf cache rm model/<repo id>
```

`rm` asks first, and takes `--dry-run`. It removes the whole repo directory,
`experts.pack` included.
