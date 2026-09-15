// Opt-in numerical trace helper for the pinned llama.cpp public C API.
// Build/run instructions live in docs/generation.md. Outputs are local artifacts.
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

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
int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: reference-generation MODEL RAW_PROMPT EXISTING_TRACE_DIR\n"); return 2; }
    ggml_backend_load_all();
    bool gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        gpu |= ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (!gpu) { std::fprintf(stderr, "Metal GPU unavailable; refusing CPU fallback\n"); return 1; }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    Trace trace{argv[3]};
    auto cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 1; cp.n_ubatch = 1;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = observe; cp.cb_eval_user_data = &trace;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 1; }
    const auto * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(128);
    int n = llama_tokenize(vocab, argv[2], std::strlen(argv[2]), tokens.data(), tokens.size(), false, true);
    int result = 0;
    if (n <= 0 || n >= 128) result = 1;
    for (int i = 0; !result && i < n; ++i) {
        trace.position = i;
        std::fprintf(stderr, "input[%d]=%d\n", i, tokens[i]);
        if (llama_decode(ctx, llama_batch_get_one(&tokens[i], 1)) || trace.failed) result = 1;
    }
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
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return result;
}
