// Opt-in EmbeddingGemma 2 rates on the second llama.cpp checkout
// (.reference/llama.cpp-embed), the method of `embeddinggemma-check MODEL
// bench`: 64 inputs of 256 tokens, then one input of 512 and one of 8192;
// each input is BOS, " the" repeated, EOS; batches of at most 8192 rows (an
// input's rows in one ubatch, as the model has no cache); one untimed
// warm-up per shape, then the median wall time of the timed runs, each
// ending when every pooled vector has been read back.
//
//   reference-embedding-bench [--cpu] MODEL
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int max_rows = 8192;

// Embeds `count` inputs of `tokens` rows in batches of at most max_rows; false on failure.
static bool embed(llama_context * ctx, llama_batch_ext * batch, const std::vector<llama_token> & input, int count) {
    const int per_batch = std::max(1, max_rows / (int) input.size());
    for (int first = 0; first < count; first += per_batch) {
        const int n = std::min(per_batch, count - first);
        llama_batch_ext_clear(batch);
        for (int s = 0; s < n; ++s) {
            for (size_t i = 0; i < input.size(); ++i) {
                const int32_t idx = llama_batch_ext_add_token(batch, s, input[i]);
                if (idx < 0) return false;
                llama_pos pos = (llama_pos) i;
                llama_batch_ext_set_pos(batch, idx, &pos);
                llama_batch_ext_set_output_embd(batch, idx, true);
            }
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) return false;
        for (int s = 0; s < n; ++s) if (!llama_get_embeddings_seq(ctx, s)) return false;
    }
    return true;
}

// Median milliseconds of `runs` timed calls after one untimed; negative on failure.
static double time_ms(llama_context * ctx, llama_batch_ext * batch, const std::vector<llama_token> & input, int count, int runs) {
    if (!embed(ctx, batch, input, count)) return -1;
    std::vector<double> times;
    for (int r = 0; r < runs; ++r) {
        const auto start = std::chrono::steady_clock::now();
        if (!embed(ctx, batch, input, count)) return -1;
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int main(int argc, char ** argv) {
    bool cpu = false;
    if (argc > 1 && std::strcmp(argv[1], "--cpu") == 0) { cpu = true; ++argv; --argc; }
    if (argc != 2) { std::fprintf(stderr, "usage: %s [--cpu] MODEL\n", argv[0]); return 2; }
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = cpu ? 0 : 99;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) return 1;
    auto cparams = llama_context_default_params();
    cparams.n_ctx = max_rows;
    cparams.n_batch = max_rows;
    cparams.n_ubatch = max_rows;
    cparams.n_seq_max = 64;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cparams.op_offload = !cpu;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token the;
    if (llama_tokenize(vocab, " the", 4, &the, 1, false, false) != 1) return 1;
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    auto input = [&](int tokens) {
        std::vector<llama_token> ids(tokens, the);
        ids.front() = llama_vocab_bos(vocab);
        ids.back() = llama_vocab_eos(vocab);
        return ids;
    };
    const double batch_ms = time_ms(ctx, batch, input(256), 64, 7);
    if (batch_ms < 0) return 1;
    std::printf("64 inputs of 256 tokens: %.1f ms, %.1f inputs/s, %.0f tokens/s\n", batch_ms, 64 * 1000 / batch_ms, 64 * 256 * 1000 / batch_ms);
    for (int tokens : {512, 8192}) {
        const double ms = time_ms(ctx, batch, input(tokens), 1, 5);
        if (ms < 0) return 1;
        std::printf("one input of %d tokens: %.1f ms, %.0f tokens/s\n", tokens, ms, tokens * 1000 / ms);
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
