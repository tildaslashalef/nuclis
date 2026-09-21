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
// `--dflash-draft DRAFT_MODEL` is Muse Glimmer's DFlash form, following the
// reference's draft-dflash driver: after the target decodes position i, the
// five target layer inputs are concatenated, projected and injected into the
// drafter's cache at i (the driver's `process()`, the graph's embd batch);
// then a 16-row noise block [next token, mask x 15] is decoded at positions
// i+1.. and its per-row hidden and greedy tokens are dumped; the noise rows
// are removed afterwards, as the driver's checkpoint reset does.
#include "llama.h"
#include "llama-ext.h" // staging: llama_set/get_embeddings_nextn for MTP
#include "ggml-backend.h"
#include <algorithm>
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

// Per-layer diagnosis of the DFlash block: the decoder graph's named
// intermediates, one row per noise token.
static bool observe_draft_layers(ggml_tensor * t, bool ask, void * opaque) {
    auto & trace = *static_cast<Trace *>(opaque);
    char suffix[64];
    int layer;
    if (std::sscanf(t->name, "l_out-%d", &layer) == 1) std::snprintf(suffix, sizeof(suffix), "l-out-%d", layer);
    else if (std::strcmp(t->name, "inp_noise_embd") == 0) std::snprintf(suffix, sizeof(suffix), "noise-embd");
    else return false;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || ggml_nelements(t) <= 0 || ggml_nelements(t) > (1 << 21)) { trace.failed = true; return false; }
    std::vector<float> values(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
    auto path = trace.directory + "/token-" + std::to_string(trace.position) + "-" + suffix + ".f32";
    if (!write_f32(path, values.data(), values.size())) trace.failed = true;
    return !trace.failed;
}

int main(int argc, char ** argv) {
    const bool mtp = argc >= 5 && std::strcmp(argv[4], "--mtp-draft") == 0;
    const bool assistant = argc == 6 && std::strcmp(argv[4], "--assistant-draft") == 0;
    const bool dflash = argc == 6 && std::strcmp(argv[4], "--dflash-draft") == 0;
    const bool draft = mtp || assistant || dflash;
    // DFLASH_DRAFT_CPU=1 leaves the drafter's model and context on the host:
    // a cross-check of the reference's own Metal block numerics.
    const bool draft_cpu = dflash && std::getenv("DFLASH_DRAFT_CPU") != nullptr;
    if (argc != 4 && !draft) {
        std::fprintf(stderr, "usage: reference-generation MODEL RAW_PROMPT EXISTING_TRACE_DIR [--mtp-draft [DRAFT_MODEL]] [--assistant-draft DRAFT_MODEL] [--dflash-draft DRAFT_MODEL]\n");
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
    if (assistant || dflash) {
        auto mp_dft = mp;
        if (draft_cpu) mp_dft.n_gpu_layers = 0;
        model_dft = llama_model_load_from_file(argv[5], mp_dft);
        if (!model_dft) { llama_model_free(model); return 1; }
    }

    Trace trace{argv[3]};
    auto cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 1; cp.n_ubatch = 1;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = dflash ? nullptr : observe; cp.cb_eval_user_data = &trace;
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
    // DFlash: the drafter's cache is filled from the target's layer inputs
    // (an embd batch per target position) and read by a 16-row noise block.
    llama_batch batch_inject = {};
    llama_batch batch_dft = {};
    llama_token mask_token = 0;
    const int32_t * target_layers = nullptr;
    uint32_t target_layer_count = 0;
    int32_t n_embd_inp = 0;
    Trace trace_dft{argv[3]};
    const bool dump_layers = std::getenv("DFLASH_DUMP_LAYERS") != nullptr;
    if (draft) {
        auto cp_mtp = cp;
        cp_mtp.cb_eval = nullptr; cp_mtp.cb_eval_user_data = nullptr;
        if (dflash) {
            cp_mtp.n_batch = 16; cp_mtp.n_ubatch = 16;
            cp_mtp.ctx_other = ctx;
            // The decoder graph's embd branch exposes the encoder output and
            // the block's final hidden through the embeddings buffer.
            cp_mtp.embeddings = true;
            if (dump_layers) { cp_mtp.cb_eval = observe_draft_layers; cp_mtp.cb_eval_user_data = &trace_dft; }
        } else {
            cp_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cp_mtp.ctx_other = assistant ? ctx : nullptr;
        }
        ctx_mtp = llama_init_from_model(model_dft, cp_mtp);
        if (!ctx_mtp) {
            fprintf(stderr, "draft context creation failed\n");
            llama_batch_free(batch);
            if (model_dft != model) llama_model_free(model_dft);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        if (dflash) {
            // The draft's block rows attend to each other and to the injected
            // prefix: non-causal, as the driver sets it.
            llama_set_causal_attn(ctx_mtp, false);
            // Unmasked: the encoder rows of the inject batch and the hidden of
            // every noise row are read back.
            llama_set_embeddings_nextn(ctx_mtp, true, /*masked*/ false);
            target_layers = llama_model_target_layer_ids(model_dft);
            target_layer_count = llama_model_target_layer_ids_n(model_dft);
            if (target_layers == nullptr || target_layer_count == 0) {
                fprintf(stderr, "draft model has no target_layers\n");
                llama_free(ctx_mtp);
                llama_batch_free(batch);
                if (model_dft != model) llama_model_free(model_dft);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            for (uint32_t k = 0; k < target_layer_count; ++k) {
                llama_set_embeddings_layer_inp(ctx, static_cast<uint32_t>(target_layers[k]), true);
            }
            mask_token = llama_vocab_mask(llama_model_get_vocab(model_dft));
            n_embd_inp = static_cast<int32_t>(target_layer_count) * n_embd;
            batch_inject = llama_batch_init(16, n_embd_inp, 1);
            batch_dft = llama_batch_init(16, 0, 1);
            greedy = std::fopen((trace.directory + "/dflash-greedy.txt").c_str(), "w");
            tokens_out = std::fopen((trace.directory + "/dflash-tokens.txt").c_str(), "w");
        } else {
            batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token)));
            if (!batch.token) {
                llama_free(ctx_mtp);
                llama_batch_free(batch);
                if (model_dft != model) llama_model_free(model_dft);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
            llama_set_embeddings_nextn(ctx_mtp, true, /*masked*/ true);
            greedy = std::fopen((trace.directory + "/mtp-greedy.txt").c_str(), "w");
            tokens_out = std::fopen((trace.directory + "/mtp-tokens.txt").c_str(), "w");
        }
        if (!greedy || !tokens_out) {
            fprintf(stderr, "cannot write draft trace files in %s\n", argv[3]);
            if (greedy) std::fclose(greedy);
            if (tokens_out) std::fclose(tokens_out);
            llama_free(ctx_mtp); llama_free(ctx); llama_batch_free(batch);
            if (model_dft != model) llama_model_free(model_dft);
            llama_model_free(model);
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
        if (dflash) {
            // process(): the target's five layer inputs at position i, fused by
            // the drafter's encoder and injected into its cache.
            batch_inject.n_tokens = 1;
            bool missing = false;
            for (uint32_t k = 0; k < target_layer_count; ++k) {
                const float * layer = llama_get_embeddings_layer_inp(ctx, static_cast<uint32_t>(target_layers[k]));
                if (layer == nullptr) { missing = true; break; }
                std::memcpy(batch_inject.embd + static_cast<size_t>(k) * n_embd, layer, sizeof(float) * static_cast<size_t>(n_embd));
            }
            if (missing) { std::fprintf(stderr, "target layer input not extracted\n"); result = 1; break; }
            batch_inject.pos[0] = i;
            batch_inject.n_seq_id[0] = 1;
            batch_inject.seq_id[0][0] = 0;
            batch_inject.logits[0] = 1;
            if (llama_decode(ctx_mtp, batch_inject)) { result = 1; break; }
            // The decoder graph's embd branch exposes its encoder output as
            // the batch's embeddings.
            const float * enc = llama_get_embeddings_ith(ctx_mtp, 0);
            if (!enc) { result = 1; break; }
            const std::string base = trace.directory + "/token-" + std::to_string(i) + "-";
            if (!write_f32(base + "encoder.f32", enc, static_cast<size_t>(n_embd))) { result = 1; break; }
            for (uint32_t k = 0; k < target_layer_count; ++k) {
                const float * layer = llama_get_embeddings_layer_inp(ctx, static_cast<uint32_t>(target_layers[k]));
                if (!write_f32(base + "inp-" + std::to_string(k) + ".f32", layer, static_cast<size_t>(n_embd))) { result = 1; break; }
            }
            if (result) break;
            if (i + 1 < n) {
                // The noise block [next token, mask x 15] at the proposal
                // position, dumped before the target decodes that token. The
                // rows are removed afterwards, as the driver's checkpoint
                // reset does, restoring the injected prefix.
                const int32_t n_block = 16;
                batch_dft.n_tokens = n_block;
                for (int32_t r = 0; r < n_block; ++r) {
                    batch_dft.token[r] = r == 0 ? tokens[i + 1] : mask_token;
                    batch_dft.pos[r] = i + 1 + r;
                    batch_dft.n_seq_id[r] = 1;
                    batch_dft.seq_id[r][0] = 0;
                    batch_dft.logits[r] = 1;
                }
                trace_dft.position = i + 1;
                if (llama_decode(ctx_mtp, batch_dft)) { result = 1; break; }
                // The block's final normed hidden, one row per noise token.
                const float * block = llama_get_embeddings_ith(ctx_mtp, 0);
                if (!block) { result = 1; break; }
                if (!write_f32(trace.directory + "/token-" + std::to_string(i + 1) + "-block-h.f32", block, static_cast<size_t>(4) * n_embd)) { result = 1; break; }
                const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
                for (int32_t r = 1; r < n_block; ++r) {
                    const float * row = llama_get_logits_ith(ctx_mtp, r);
                    int best = 0;
                    for (int k = 1; k < vocab_size; ++k) if (row[k] > row[best]) best = k;
                    std::fprintf(greedy, "%d\n", best);
                }
                std::fflush(greedy);
                llama_memory_seq_rm(llama_get_memory(ctx_mtp), 0, i + 1, -1);
            }
            continue;
        }
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
    if (!result && !dflash) {
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
    if (dflash) { llama_batch_free(batch_inject); llama_batch_free(batch_dft); }
    if (model_dft != model) llama_model_free(model_dft);
    llama_model_free(model); llama_backend_free();
    return result;
}
