// Opt-in oracle for the vision projectors, against the pinned llama.cpp's
// libmtmd. Build/run instructions live in docs/reference/vision.md; outputs
// are local artifacts under .zig-cache/, copied into tests/fixtures/vision/
// only as a reviewed fixture commit.
//
//   reference-vision MODEL MMPROJ IMAGE PROMPT OUTDIR [--cpu-vision]
//                    [--n-predict N] [--trace NAME[,NAME...]]
//
// PROMPT is the rendered prompt text (special tokens parsed) with the
// marker <__media__> where the image goes. Written to OUTDIR:
//   prompt-tokens.json   the text tokens per chunk, with the image chunk's
//                        n_tokens, grid, n_pos and per-token (t, x, y)
//   image-embd.f32       the projector's rows: n_tokens × n_embd_inp
//   logits.f32           the last prompt position's logits
//   greedy.txt           N greedy tokens after the prompt (ids, one per line)
//   <name>.f32           each --trace node of the projector graph (F32/F16)
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

static bool write_f32(const std::string & path, const float * data, size_t count) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(data, sizeof(float), count, f) == count;
    if (std::fclose(f)) ok = false;
    return ok;
}

struct Trace { std::string directory; std::set<std::string> names; bool list = false; bool failed = false; };

static bool trace_cb(ggml_tensor * t, bool ask, void * opaque) {
    auto & trace = *static_cast<Trace *>(opaque);
    if (ask) {
        if (trace.list) std::fprintf(stderr, "node %s [%lld, %lld, %lld, %lld] %s\n", t->name,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type));
        return trace.names.count(t->name) > 0;
    }
    if (trace.names.count(t->name) == 0) return true;
    size_t n = (size_t) ggml_nelements(t);
    std::vector<float> values(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, values.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(n);
        ggml_backend_tensor_get(t, half.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++) values[i] = ggml_fp16_to_fp32(half[i]);
    } else {
        std::fprintf(stderr, "trace: %s has type %s, not dumped\n", t->name, ggml_type_name(t->type));
        return true;
    }
    std::string name = t->name;
    for (auto & c : name) if (c == '/' || c == ' ') c = '_';
    if (!write_f32(trace.directory + "/" + name + ".f32", values.data(), n)) trace.failed = true;
    std::fprintf(stderr, "trace %s [%lld, %lld, %lld, %lld]\n", t->name,
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s MODEL MMPROJ IMAGE PROMPT OUTDIR [--cpu-vision] [--n-predict N] [--trace NAMES] [--list-nodes]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1], mmproj_path = argv[2], image_path = argv[3], prompt = argv[4], outdir = argv[5];
    bool cpu_vision = false;
    int n_predict = 8;
    Trace trace; trace.directory = outdir;
    for (int i = 6; i < argc; i++) {
        if (!std::strcmp(argv[i], "--cpu-vision")) cpu_vision = true;
        else if (!std::strcmp(argv[i], "--n-predict") && i + 1 < argc) n_predict = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--list-nodes")) trace.list = true;
        else if (!std::strcmp(argv[i], "--trace") && i + 1 < argc) {
            std::string s = argv[++i];
            size_t start = 0;
            while (start <= s.size()) {
                size_t comma = s.find(',', start);
                if (comma == std::string::npos) comma = s.size();
                if (comma > start) trace.names.insert(s.substr(start, comma - start));
                start = comma + 1;
            }
        } else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 4096;
    cparams.n_ubatch = 1024;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * lctx = llama_init_from_model(model, cparams);
    if (!lctx) { std::fprintf(stderr, "context init failed\n"); return 1; }

    mtmd_context_params vparams = mtmd_context_params_default();
    vparams.use_gpu = !cpu_vision;
    vparams.n_threads = 8;
    vparams.print_timings = false;
    vparams.warmup = false;
    vparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (!trace.names.empty() || trace.list) { vparams.cb_eval = trace_cb; vparams.cb_eval_user_data = &trace; }
    mtmd_context * vctx = mtmd_init_from_file(mmproj_path.c_str(), model, vparams);
    if (!vctx) { std::fprintf(stderr, "mmproj load failed\n"); return 1; }
    std::fprintf(stderr, "mrope=%d non_causal=%d marker=%s\n", mtmd_decode_use_mrope(vctx), mtmd_decode_use_non_causal(vctx, nullptr), mtmd_default_marker());

    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(vctx, image_path.c_str(), false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) { std::fprintf(stderr, "image load failed\n"); return 1; }
    std::fprintf(stderr, "image %ux%u\n", mtmd_bitmap_get_nx(bw.bitmap), mtmd_bitmap_get_ny(bw.bitmap));

    mtmd_input_text text; text.text = prompt.c_str(); text.text_len = prompt.size(); text.add_special = false; text.parse_special = true;
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[1] = { bw.bitmap };
    if (mtmd_tokenize(vctx, chunks, &text, bitmaps, 1) != 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }

    const int n_embd_inp = llama_model_n_embd_inp(model);
    FILE * pt = std::fopen((outdir + "/prompt-tokens.json").c_str(), "w");
    if (!pt) { std::fprintf(stderr, "cannot write prompt-tokens.json\n"); return 1; }
    std::fprintf(pt, "{\"n_embd_inp\": %d, \"chunks\": [", n_embd_inp);
    llama_pos n_past = 0;
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (i) std::fprintf(pt, ",");
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n);
            std::fprintf(pt, "{\"type\": \"text\", \"tokens\": [");
            for (size_t k = 0; k < n; k++) std::fprintf(pt, "%s%d", k ? ", " : "", toks[k]);
            std::fprintf(pt, "]}");
        } else if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const mtmd_image_tokens * img = mtmd_input_chunk_get_tokens_image(chunk);
            size_t n = mtmd_image_tokens_get_n_tokens(img);
            llama_pos n_pos = mtmd_image_tokens_get_n_pos(img);
            std::fprintf(pt, "{\"type\": \"image\", \"n_tokens\": %zu, \"n_pos\": %d, \"pos_0\": %d, \"positions\": [", n, n_pos, n_past);
            for (size_t k = 0; k < n; k++) {
                mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(img, n_past, k);
                std::fprintf(pt, "%s[%u, %u, %u]", k ? ", " : "", p.t, p.x, p.y);
            }
            std::fprintf(pt, "]}");
            if (mtmd_encode_chunk(vctx, chunk) != 0) { std::fprintf(stderr, "encode failed\n"); return 1; }
            const float * embd = mtmd_get_output_embd(vctx);
            if (!write_f32(outdir + "/image-embd.f32", embd, n * (size_t) n_embd_inp)) { std::fprintf(stderr, "cannot write image-embd.f32\n"); return 1; }
            std::fprintf(stderr, "image chunk: %zu tokens, n_pos %d, %d wide\n", n, n_pos, n_embd_inp);
        } else {
            std::fprintf(stderr, "unexpected chunk type\n"); return 1;
        }
        llama_pos new_n_past = n_past;
        if (mtmd_helper_eval_chunk_single(vctx, lctx, chunk, n_past, 0, 1024, i == n_chunks - 1, &new_n_past) != 0) {
            std::fprintf(stderr, "eval of chunk %zu failed\n", i); return 1;
        }
        n_past = new_n_past;
    }
    std::fprintf(pt, "], \"n_past\": %d}\n", n_past);
    std::fclose(pt);
    if (trace.failed) { std::fprintf(stderr, "trace write failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(lctx, -1);
    if (!write_f32(outdir + "/logits.f32", logits, (size_t) n_vocab)) { std::fprintf(stderr, "cannot write logits.f32\n"); return 1; }

    FILE * g = std::fopen((outdir + "/greedy.txt").c_str(), "w");
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (int step = 0; step < n_predict; step++) {
        const float * l = llama_get_logits_ith(lctx, -1);
        int best = 0;
        for (int v = 1; v < n_vocab; v++) if (l[v] > l[best]) best = v;
        char piece[256];
        int len = llama_token_to_piece(llama_model_get_vocab(model), best, piece, sizeof(piece) - 1, 0, true);
        if (len < 0) len = 0;
        piece[len] = 0;
        std::fprintf(g, "%d\n", best);
        std::fprintf(stderr, "greedy %d: %d %s\n", step, best, piece);
        batch.n_tokens = 1; batch.token[0] = best; batch.pos[0] = n_past++; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
        if (llama_decode(lctx, batch)) { std::fprintf(stderr, "decode failed\n"); return 1; }
    }
    std::fclose(g);
    llama_batch_free(batch);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(vctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
