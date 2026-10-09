// Opt-in EmbeddingGemma 2 oracle on the second llama.cpp checkout
// (.reference/llama.cpp-embed, docs/benchmarks/llama-cpp.md § The third
// oracle). Reads tests/fixtures/embeddinggemma-inputs/inputs.json, builds each
// case's parts with mtmd_tokenize_from_parts, and runs the whole input as ONE
// mixed batch (token rows and projector rows together): the model has no KV
// cache, so separate decode calls would not attend to each other.
//
//   reference-embedding [--cpu] [--text-only] MODEL MMPROJ INPUTS VECTORS_PREFIX [TRACE_ROOT]
//
// --cpu keeps every operation on ggml's CPU backend (no layers or ops
// offloaded): on an F32 file its matmuls keep F32 activations, where Metal's
// batched matmul stages them as half (bf16 for BF16 weights). --text-only
// skips the cases with media.
//
// Writes VECTORS_PREFIX.f32 (768 unit-normalized F32 per case, case order) and
// VECTORS_PREFIX.json (per case its ids run-length encoded, a media chunk's
// rows shown as its placeholder id). With TRACE_ROOT, each case in the
// file's "traced" list also gets TRACE_ROOT/embeddinggemma-<case>/ with
// shapes.json: a text-only case holds inp_scaled, inp_per_layer, every
// l_out-<il>, result_norm and result_embd; a case with media (whose layers the
// text cases already pin) holds inp_scaled, l_out-0, l_out-23, and per media
// chunk media-<k>.f32, its projector rows.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// The placeholder ids the processor writes where media rows go
// (Google's config.json `image_token_id`, `audio_token_id`).
static constexpr llama_token image_token = 258880;
static constexpr llama_token audio_token = 258881;

struct Trace {
    std::string directory;
    json shapes = json::object();
    bool media = false;
    bool failed = false;
};

static bool write_f32(const std::string & path, const float * data, size_t count) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(data, sizeof(float), count, f) == count;
    if (std::fclose(f)) ok = false;
    return ok;
}

static bool traced_name(const char * name, bool media) {
    int layer;
    if (std::strcmp(name, "inp_scaled") == 0) return true;
    if (std::sscanf(name, "l_out-%d", &layer) == 1) return !media || layer == 0 || layer == 23;
    return !media && (std::strcmp(name, "inp_per_layer") == 0 || std::strcmp(name, "result_norm") == 0 ||
                      std::strcmp(name, "result_embd") == 0);
}

static bool observe(ggml_tensor * t, bool ask, void * opaque) {
    auto * trace = static_cast<Trace *>(opaque);
    if (!trace || trace->directory.empty() || !traced_name(t->name, trace->media)) return !ask;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { trace->failed = true; return false; }
    std::vector<float> values(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
    if (!write_f32(trace->directory + "/" + t->name + ".f32", values.data(), values.size())) trace->failed = true;
    trace->shapes[t->name] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    return !trace->failed;
}

static void push_run(json & runs, llama_token id, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!runs.empty() && runs.back().is_array() && runs.back()[0] == id) runs.back()[1] = runs.back()[1].get<size_t>() + 1;
        else if (!runs.empty() && runs.back().is_number() && runs.back() == id) runs.back() = json::array({id, 2});
        else runs.push_back(id);
    }
}

int main(int argc, char ** argv) {
    bool cpu = false, text_only = false;
    while (argc > 1 && std::strncmp(argv[1], "--", 2) == 0) {
        if (std::strcmp(argv[1], "--cpu") == 0) cpu = true;
        else if (std::strcmp(argv[1], "--text-only") == 0) text_only = true;
        else return 2;
        ++argv;
        --argc;
    }
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr, "usage: %s [--cpu] [--text-only] MODEL MMPROJ INPUTS VECTORS_PREFIX [TRACE_ROOT]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1], mmproj_path = argv[2], inputs_path = argv[3], prefix = argv[4];
    const std::string trace_root = argc == 6 ? argv[5] : "";
    std::ifstream in(inputs_path);
    const json spec = json::parse(in);
    // Media paths in the inputs file are relative to the repository root.
    const std::string root = inputs_path.substr(0, inputs_path.rfind("tests/fixtures/"));

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = cpu ? 0 : 99;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) return 1;

    Trace trace;
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 8192;
    cparams.n_batch = 8192;
    cparams.n_ubatch = 8192; // a bidirectional input must be one ubatch
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.op_offload = !cpu;
    cparams.cb_eval = observe;
    cparams.cb_eval_user_data = &trace;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 1;

    auto vparams = mtmd_context_params_default();
    vparams.use_gpu = !cpu;
    vparams.print_timings = false;
    vparams.warmup = false;
    // Google's processor budget; clip.cpp otherwise sizes toward its 1120 cap.
    vparams.image_max_tokens = spec.value("image_tokens", 280);
    mtmd_context * vision = mtmd_init_from_file(mmproj_path.c_str(), model, vparams);
    if (!vision) return 1;

    const int n_embd = llama_model_n_embd_inp(model);
    const int n_out = llama_model_n_embd_out(model);
    std::vector<std::string> traced = spec.value("traced", std::vector<std::string>{});
    std::vector<float> vectors;
    json records = json::array();
    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    for (const auto & c : spec["cases"]) {
        const std::string id = c["id"];
        bool has_media = false;
        for (const auto & p : c["parts"]) has_media = has_media || !p.contains("text");
        if (text_only && has_media) continue;
        trace.directory.clear();
        trace.shapes = json::object();
        trace.media = false;
        for (const auto & p : c["parts"]) trace.media = trace.media || !p.contains("text");
        if (!trace_root.empty() && std::find(traced.begin(), traced.end(), id) != traced.end()) {
            trace.directory = trace_root + "/embeddinggemma-" + id;
            std::string cmd = "mkdir -p '" + trace.directory + "'";
            if (std::system(cmd.c_str()) != 0) return 1;
        }

        // Parts in order: text, media as bitmaps.
        std::vector<std::string> texts;
        texts.reserve(c["parts"].size());
        std::vector<mtmd_input_text> text_parts(c["parts"].size());
        std::vector<mtmd_bitmap *> bitmaps;
        std::vector<mtmd_input_part> parts(c["parts"].size());
        std::vector<const mtmd_input_part *> part_ptrs;
        for (size_t i = 0; i < c["parts"].size(); ++i) {
            const auto & p = c["parts"][i];
            if (p.contains("text")) {
                texts.push_back(p["text"].get<std::string>());
                // parse_special: the HF tokenizer turns special-token text (a literal `<bos>`) into the token.
                text_parts[i] = {texts.back().c_str(), texts.back().size(), false, true};
                parts[i] = {&text_parts[i], nullptr};
            } else {
                const std::string path = root + (p.contains("image") ? p["image"] : p["audio"]).get<std::string>();
                auto w = mtmd_helper_bitmap_init_from_file(vision, path.c_str(), false, mtmd_helper_init_opt_default());
                if (!w.bitmap) { std::fprintf(stderr, "%s: cannot load %s\n", id.c_str(), path.c_str()); return 1; }
                bitmaps.push_back(w.bitmap);
                parts[i] = {nullptr, w.bitmap};
            }
            part_ptrs.push_back(&parts[i]);
        }
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        if (mtmd_tokenize_from_parts(vision, chunks, part_ptrs.data(), part_ptrs.size(), /*add_special*/ true) != 0) {
            std::fprintf(stderr, "%s: tokenize failed\n", id.c_str());
            return 1;
        }

        // Encode every media chunk first; keep its rows until the batch has run.
        std::vector<std::vector<float>> media;
        llama_batch_ext_clear(batch);
        json runs = json::array();
        llama_pos pos = 0;
        auto add_row = [&](int32_t idx) {
            llama_batch_ext_set_pos(batch, idx, &pos);
            llama_batch_ext_set_output_embd(batch, idx, true);
            ++pos;
        };
        for (size_t k = 0; k < mtmd_input_chunks_size(chunks); ++k) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, k);
            const auto type = mtmd_input_chunk_get_type(chunk);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
                for (size_t i = 0; i < n; ++i) {
                    int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
                    if (idx < 0) return 1;
                    add_row(idx);
                    push_run(runs, tokens[i], 1);
                }
                continue;
            }
            if (mtmd_encode_chunk(vision, chunk) != 0) { std::fprintf(stderr, "%s: encode failed\n", id.c_str()); return 1; }
            const size_t n = mtmd_input_chunk_get_n_tokens(chunk);
            const float * rows = mtmd_get_output_embd(vision);
            media.emplace_back(rows, rows + n * n_embd);
            if (!trace.directory.empty()) {
                auto name = "media-" + std::to_string(media.size() - 1);
                if (!write_f32(trace.directory + "/" + name + ".f32", media.back().data(), media.back().size())) return 1;
                trace.shapes[name] = {n_embd, n, 1, 1};
            }
            for (size_t i = 0; i < n; ++i) {
                int32_t idx = llama_batch_ext_add_embd(batch, 0, {media.back().data() + i * n_embd, 1, (size_t) n_embd});
                if (idx < 0) return 1;
                add_row(idx);
            }
            push_run(runs, type == MTMD_INPUT_CHUNK_TYPE_AUDIO ? audio_token : image_token, n);
        }
        mtmd_input_chunks_free(chunks);
        for (auto * b : bitmaps) mtmd_bitmap_free(b);

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0 || trace.failed) {
            std::fprintf(stderr, "%s: decode failed\n", id.c_str());
            return 1;
        }
        const float * pooled = llama_get_embeddings_seq(ctx, 0);
        if (!pooled) return 1;
        double norm = 0;
        for (int i = 0; i < n_out; ++i) norm += (double) pooled[i] * pooled[i];
        norm = std::sqrt(norm);
        for (int i = 0; i < n_out; ++i) vectors.push_back((float) (pooled[i] / norm));
        records.push_back({{"id", id}, {"tokens", pos}, {"ids", runs}, {"pooled_norm", norm}});
        if (!trace.directory.empty()) std::ofstream(trace.directory + "/shapes.json") << trace.shapes.dump(1) << "\n";
        std::fprintf(stderr, "%-28s %5d tokens\n", id.c_str(), (int) pos);
    }

    if (!write_f32(prefix + ".f32", vectors.data(), vectors.size())) return 1;
    json meta = {{"source", model_path.substr(model_path.rfind('/') + 1)},
                 {"mmproj", mmproj_path.substr(mmproj_path.rfind('/') + 1)},
                 {"dimensions", n_out},
                 {"cases", records}};
    std::ofstream(prefix + ".json") << meta.dump(1) << "\n";

    llama_batch_ext_free(batch);
    mtmd_free(vision);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
