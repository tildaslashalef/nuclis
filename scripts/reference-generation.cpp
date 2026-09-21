// Opt-in numerical trace helper for the pinned llama.cpp API.
// Build/run instructions live in docs/reference/generation.md. Outputs are
// local artifacts. With --mtp-draft it additionally opens the file as an MTP
// context and dumps, per prompt position, the prediction block's pair inputs
// (target h_{p-1}, token x_p), its greedy draft token, and its output h — the
// pinned oracle for the Qwen draft head (docs/reference/speculative-decoding.md).
// `--assistant-draft DRAFT_MODEL` is the Gemma 4 companion form: it loads the
// gemma4-assistant file, points its context at the target (`ctx_other`) so the
// head reads the target's layer-46/47 caches, and dumps the same rows for the
// query position *before* the target decodes that token, which is where
// `propose` runs.
#include "llama.h"
#include "llama-ext.h" // staging: llama_set/get_embeddings_nextn for MTP
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Trace { std::string directory; int position = 0; bool failed = false; };
static bool observe(ggml_tensor * t, bool ask, void * opaque) {
    auto & trace = *static_cast<Trace *>(opaque);
    int layer;
    if (std::sscanf(t->name, "l_out-%d", &layer) != 1) return false;
    if (ask) return true;
    // Any F32 width: the comparison script checks the geometry it expects.
    if (t->type != GGML_TYPE_F32 || ggml_nelements(t) <= 0 || ggml_nelements(t) > (1 << 20)) { trace.failed = true; return false; }
    std::vector<float> values(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
    auto path = trace.directory + "/token-" + std::to_string(trace.position) + "-layer-" + std::to_string(layer) + ".f32";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { trace.failed = true; return false; }
    if (std::fwrite(values.data(), sizeof(float), values.size(), f) != values.size()) trace.failed = true;
    if (std::fclose(f)) trace.failed = true;
    return !trace.failed;
}

static bool write_f32(const std::string & path, const float * data, size_t count) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(data, sizeof(float), count, f) == count;
    if (std::fclose(f)) ok = false;
    return ok;
}

int main(int argc, char ** argv) {
    const bool mtp = argc >= 5 && std::strcmp(argv[4], "--mtp-draft") == 0;
    const bool assistant = argc == 6 && std::strcmp(argv[4], "--assistant-draft") == 0;
    const bool draft = mtp || assistant;
    if (argc != 4 && !draft) {
        std::fprintf(stderr, "usage: reference-generation MODEL RAW_PROMPT EXISTING_TRACE_DIR [--mtp-draft [DRAFT_MODEL]]\n");
        return 2;
    }
    ggml_backend_load_all();
    bool gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        gpu |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (!gpu) { std::fprintf(stderr, "Metal GPU unavailable; refusing CPU fallback\n"); return 1; }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.load_mtp = mtp; // the MTP tensors are skipped unless asked for
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    // The Gemma assistant is a second model file whose context shares the
    // target's KV cache layers.
    auto * model_dft = model;
    if (assistant) {
        model_dft = llama_model_load_from_file(argv[5], mp);
        if (!model_dft) { llama_model_free(model); return 1; }
    }

    Trace trace{argv[3]};
    auto cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 1; cp.n_ubatch = 1;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = observe; cp.cb_eval_user_data = &trace;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 1; }
    if (draft) llama_set_embeddings_nextn(ctx, true, /*masked*/ false);

    // The prediction block runs in its own context over the same file with its
    // own attention cache; the target exposes each position's hidden.
    const int32_t n_embd = llama_model_n_embd_out(model);
    llama_context * ctx_mtp = nullptr;
    // The MTP batch carries a token id and an h row per position: llama_batch_init
    // allocates one of the two, so the token array is added by hand.
    llama_batch batch = llama_batch_init(1, n_embd, 1);
    FILE * greedy = nullptr;
    FILE * tokens_out = nullptr;
    std::vector<float> h_prev(static_cast<size_t>(n_embd), 0.0f);
    if (draft) {
        batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token)));
        auto cp_mtp = cp;
        cp_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp_mtp.ctx_other = assistant ? ctx : nullptr;
        cp_mtp.cb_eval = nullptr; cp_mtp.cb_eval_user_data = nullptr;
        ctx_mtp = llama_init_from_model(model_dft, cp_mtp);
        if (!batch.token || !ctx_mtp) {
            fprintf(stderr, "MTP context or batch allocation failed\n");
            llama_batch_free(batch);
            llama_model_free(model);
            return 1;
        }
        llama_set_embeddings_nextn(ctx_mtp, true, /*masked*/ true);
        greedy = std::fopen((trace.directory + "/mtp-greedy.txt").c_str(), "w");
        tokens_out = std::fopen((trace.directory + "/mtp-tokens.txt").c_str(), "w");
        if (!greedy || !tokens_out) {
            fprintf(stderr, "cannot write MTP trace files in %s\n", argv[3]);
            if (greedy) std::fclose(greedy);
            if (tokens_out) std::fclose(tokens_out);
            llama_free(ctx_mtp); llama_free(ctx); llama_batch_free(batch); llama_model_free(model);
            return 1;
        }
    }

    const auto * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(128);
    int n = llama_tokenize(vocab, argv[2], std::strlen(argv[2]), tokens.data(), tokens.size(), false, true);
    int result = 0;
    if (n <= 0 || n >= 128) result = 1;
    for (int i = 0; !result && i < n; ++i) {
        if (tokens_out) std::fprintf(tokens_out, "%d\n", tokens[i]);
        trace.position = i;
        std::fprintf(stderr, "input[%d]=%d\n", i, tokens[i]);
        // The assistant's head never writes KV: it is a pure reader of the
        // target's caches, so its query sits at the position of the token to
        // propose, before that token is in the target cache. That is the
        // driver's `draft()` moment, and it is what the native `propose`
        // reproduces.
        if (assistant && i > 0) {
            batch.n_tokens = 1;
            batch.token[0] = tokens[i];
            std::memcpy(batch.embd, h_prev.data(), sizeof(float) * static_cast<size_t>(n_embd));
            batch.pos[0] = i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx_mtp, batch)) { result = 1; break; }
            const float * logits = llama_get_logits_ith(ctx_mtp, 0);
            const int size = llama_vocab_n_tokens(vocab);
            int best = 0;
            for (int k = 1; k < size; ++k) if (logits[k] > logits[best]) best = k;
            const float * h_mtp = llama_get_embeddings_nextn(ctx_mtp);
            if (!h_mtp) { result = 1; break; }
            const std::string base = trace.directory + "/token-" + std::to_string(i) + "-mtp-";
            if (!write_f32(base + "hprev.f32", h_prev.data(), h_prev.size())) { result = 1; break; }
            if (!write_f32(base + "h.f32", h_mtp, static_cast<size_t>(n_embd))) { result = 1; break; }
            std::fprintf(greedy, "%d\n", best);
            std::fflush(greedy);
        }
        if (llama_decode(ctx, llama_batch_get_one(&tokens[i], 1)) || trace.failed) { result = 1; break; }
        if (!draft) continue;
        // Position i pairs the token with the previous position's target
        // hidden; the first position pairs with zeros, as the driver does.
        const float * h_cur = llama_get_embeddings_nextn(ctx);
        if (!h_cur) { result = 1; break; }
        std::vector<float> h_now(h_cur, h_cur + n_embd);
        if (mtp) {
            batch.n_tokens = 1;
            batch.token[0] = tokens[i];
            std::memcpy(batch.embd, h_prev.data(), sizeof(float) * static_cast<size_t>(n_embd));
            batch.pos[0] = i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx_mtp, batch)) { result = 1; break; }
            const float * logits = llama_get_logits_ith(ctx_mtp, 0);
            const int size = llama_vocab_n_tokens(vocab);
            int best = 0;
            for (int k = 1; k < size; ++k) if (logits[k] > logits[best]) best = k;
            const float * h_mtp = llama_get_embeddings_nextn(ctx_mtp);
            if (!h_mtp) { result = 1; break; }
            const std::string base = trace.directory + "/token-" + std::to_string(i) + "-mtp-";
            if (!write_f32(base + "hprev.f32", h_prev.data(), h_prev.size())) { result = 1; break; }
            if (!write_f32(base + "h.f32", h_mtp, static_cast<size_t>(n_embd))) { result = 1; break; }
            std::fprintf(greedy, "%d\n", best);
            std::fflush(greedy);
        }
        h_prev = std::move(h_now);
    }
    if (!result && draft) std::printf("draft trace written to %s\n", argv[3]);
    if (!result) {
        auto * logits = llama_get_logits_ith(ctx, -1);
        int size = llama_vocab_n_tokens(vocab), best = 0;
        for (int i = 1; i < size; ++i) if (logits[i] > logits[best]) best = i;
        auto path = trace.directory + "/logits.f32";
        FILE * f = std::fopen(path.c_str(), "wb");
        if (!f) result = 1;
        else { if (std::fwrite(logits, sizeof(float), size, f) != size) result = 1; if (std::fclose(f)) result = 1; }
        std::printf("greedy_token=%d\n", best);
    }
    if (greedy) std::fclose(greedy);
    if (tokens_out) std::fclose(tokens_out);
    if (ctx_mtp) llama_free(ctx_mtp);
    llama_free(ctx); llama_batch_free(batch);
    if (model_dft != model) llama_model_free(model_dft);
    llama_model_free(model); llama_backend_free();
    return result;
}
