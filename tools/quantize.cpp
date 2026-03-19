// diffuse-quantize: Convert F16/F32 GGUF to quantized GGUF
//
// Supported target types:
//   q8_0   — 8-bit quantization (~8.5 GB for 8B model)
//   q4_k_m — mixed 4-bit K-quant (~4.5 GB for 8B model, best quality/size)
//
// Usage: diffuse-quantize input.gguf output.gguf q4_k_m

#include <ggml.h>
#include <gguf.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// ── Tensor classification ─────────────────────────────────────
// Determines what quantization type to use for each tensor based on its name.

enum tensor_class {
    TC_NORM,       // Norm weights — always F32
    TC_EMBED,      // Token embeddings — keep F16
    TC_OUTPUT,     // lm_head — Q6_K (needs precision for vocabulary)
    TC_ATTN,       // Attention weights (Q/K/V/O)
    TC_FFN,        // FFN weights (gate/up/down)
    TC_FFN_EDGE,   // FFN in first or last layer (higher precision)
};

static tensor_class classify_tensor(const char * name, int n_layers) {
    // Norms are always kept in F32
    if (strstr(name, "_norm") || strstr(name, "norm.")) return TC_NORM;

    // Token embeddings
    if (strstr(name, "token_embd") || strstr(name, "embed")) return TC_EMBED;

    // Extract layer number for edge detection
    int layer = -1;
    const char * blk = strstr(name, "blk.");
    if (blk) {
        layer = atoi(blk + 4);
    }

    // Attention weights (check before output to catch attn_output)
    if (strstr(name, "attn_q") || strstr(name, "attn_k") ||
        strstr(name, "attn_v") || strstr(name, "attn_output")) {
        return TC_ATTN;
    }

    // FFN weights
    if (strstr(name, "ffn_gate") || strstr(name, "ffn_up") || strstr(name, "ffn_down")) {
        if (layer == 0 || layer == n_layers - 1) return TC_FFN_EDGE;
        return TC_FFN;
    }

    // Output head (lm_head / output.weight — not inside a layer block)
    if (strstr(name, "output.weight") || strstr(name, "lm_head")) return TC_OUTPUT;

    // Default: treat as attention weight
    return TC_ATTN;
}

// ── Quantization type assignment ──────────────────────────────

struct quant_scheme {
    std::string name;
    ggml_type norm_type;
    ggml_type embed_type;
    ggml_type output_type;
    ggml_type attn_type;
    ggml_type ffn_type;
    ggml_type ffn_edge_type;
};

static quant_scheme get_scheme(const std::string & type_str) {
    if (type_str == "q8_0") {
        return {"Q8_0", GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0,
                GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0};
    }
    if (type_str == "q4_k_m" || type_str == "q4_k") {
        return {"Q4_K_M", GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q6_K,
                GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K};
    }
    if (type_str == "q4_0") {
        return {"Q4_0", GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0,
                GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0};
    }
    if (type_str == "q6_k") {
        return {"Q6_K", GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q6_K,
                GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K};
    }
    fprintf(stderr, "Unknown quantization type: %s\n", type_str.c_str());
    fprintf(stderr, "Supported: q8_0, q4_k_m, q4_0, q6_k\n");
    exit(1);
    return {};  // unreachable
}

static ggml_type target_type_for(tensor_class tc, const quant_scheme & scheme) {
    switch (tc) {
        case TC_NORM:     return scheme.norm_type;
        case TC_EMBED:    return scheme.embed_type;
        case TC_OUTPUT:   return scheme.output_type;
        case TC_ATTN:     return scheme.attn_type;
        case TC_FFN:      return scheme.ffn_type;
        case TC_FFN_EDGE: return scheme.ffn_edge_type;
    }
    return GGML_TYPE_F32;
}

static const char * tc_name(tensor_class tc) {
    switch (tc) {
        case TC_NORM:     return "norm";
        case TC_EMBED:    return "embed";
        case TC_OUTPUT:   return "output";
        case TC_ATTN:     return "attn";
        case TC_FFN:      return "ffn";
        case TC_FFN_EDGE: return "ffn_edge";
    }
    return "?";
}

// ── Main ──────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <type>\n", argv[0]);
        fprintf(stderr, "\nTypes: q8_0, q4_k_m, q4_0, q6_k\n");
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s llada-8b-f16.gguf llada-8b-q4km.gguf q4_k_m\n", argv[0]);
        return 1;
    }

    const char * input_path  = argv[1];
    const char * output_path = argv[2];
    const std::string type_str = argv[3];
    const quant_scheme scheme = get_scheme(type_str);

    fprintf(stderr, "Quantizing: %s → %s [%s]\n", input_path, output_path, scheme.name.c_str());

    // ── Load input GGUF ───────────────────────────────────────
    struct ggml_context * src_ctx = nullptr;
    struct gguf_init_params params = { false, &src_ctx };
    struct gguf_context * src_gctx = gguf_init_from_file(input_path, params);
    if (!src_gctx) {
        fprintf(stderr, "Failed to open input GGUF: %s\n", input_path);
        return 1;
    }

    const int n_tensors = gguf_get_n_tensors(src_gctx);
    fprintf(stderr, "Input: %d tensors\n", n_tensors);

    // Get n_layers for edge detection
    int n_layers = 32;  // default
    int64_t key_id = gguf_find_key(src_gctx, "diffuse.block_count");
    if (key_id >= 0) {
        n_layers = (int)gguf_get_val_u32(src_gctx, key_id);
    }
    fprintf(stderr, "Layers: %d\n", n_layers);

    // ── Create output GGUF ────────────────────────────────────
    struct gguf_context * dst_gctx = gguf_init_empty();

    // Copy all metadata from source
    gguf_set_kv(dst_gctx, src_gctx);

    // ── Process each tensor ───────────────────────────────────
    // Keep all data buffers alive until gguf_write_to_file completes
    std::vector<std::vector<float>>   f32_buffers;
    std::vector<std::vector<uint8_t>> quant_buffers;
    std::vector<struct ggml_context *> tmp_contexts;

    size_t total_src_bytes = 0;
    size_t total_dst_bytes = 0;
    int n_quantized = 0;
    int n_kept = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(src_gctx, i);
        struct ggml_tensor * src_tensor = ggml_get_tensor(src_ctx, name);
        if (!src_tensor) {
            fprintf(stderr, "Warning: tensor '%s' not found in context, skipping\n", name);
            continue;
        }

        const ggml_type src_type = src_tensor->type;
        const int64_t nelements = ggml_nelements(src_tensor);
        const size_t src_nbytes = ggml_nbytes(src_tensor);
        total_src_bytes += src_nbytes;

        // Classify and determine target type
        tensor_class tc = classify_tensor(name, n_layers);
        ggml_type dst_type = target_type_for(tc, scheme);

        // Don't "quantize" to a larger or same type
        if (dst_type == src_type || (!ggml_is_quantized(dst_type) && dst_type >= src_type)) {
            // Keep as-is
            gguf_add_tensor(dst_gctx, src_tensor);
            total_dst_bytes += src_nbytes;
            n_kept++;
            fprintf(stderr, "  [%3d/%d] %-50s  %6s → %6s  (keep, %s)\n",
                    i + 1, n_tensors, name,
                    ggml_type_name(src_type), ggml_type_name(src_type),
                    tc_name(tc));
            continue;
        }

        // Need to quantize: first dequant to F32
        f32_buffers.emplace_back(nelements);
        std::vector<float> & f32_data = f32_buffers.back();

        if (src_type == GGML_TYPE_F32) {
            memcpy(f32_data.data(), src_tensor->data, src_nbytes);
        } else if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row(
                (const ggml_fp16_t *)src_tensor->data,
                f32_data.data(), nelements);
        } else if (src_type == GGML_TYPE_BF16) {
            // BF16 → F32
            const ggml_bf16_t * bf16_data = (const ggml_bf16_t *)src_tensor->data;
            for (int64_t j = 0; j < nelements; j++) {
                f32_data[j] = ggml_bf16_to_fp32(bf16_data[j]);
            }
        } else {
            fprintf(stderr, "Error: unsupported source type %s for tensor %s\n",
                    ggml_type_name(src_type), name);
            return 1;
        }

        // For norm/embed that target F32, just write F32
        if (dst_type == GGML_TYPE_F32) {
            struct ggml_init_params tmp_params = {
                /*.mem_size   =*/ ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            struct ggml_context * tmp_ctx = ggml_init(tmp_params);
            tmp_contexts.push_back(tmp_ctx);

            struct ggml_tensor * dst_tensor = ggml_new_tensor(
                tmp_ctx, GGML_TYPE_F32, ggml_n_dims(src_tensor), src_tensor->ne);
            ggml_set_name(dst_tensor, name);
            dst_tensor->data = f32_data.data();

            gguf_add_tensor(dst_gctx, dst_tensor);
            gguf_set_tensor_data(dst_gctx, name, f32_data.data());

            size_t dst_nbytes = nelements * sizeof(float);
            total_dst_bytes += dst_nbytes;
            n_kept++;

            fprintf(stderr, "  [%3d/%d] %-50s  %6s → %6s  (to f32, %s)\n",
                    i + 1, n_tensors, name,
                    ggml_type_name(src_type), "F32",
                    tc_name(tc));
            continue;
        }

        // Quantize F32 → target type
        // Figure out dimensions for ggml_quantize_chunk
        int64_t nrows = 1;
        int64_t n_per_row = nelements;
        if (ggml_n_dims(src_tensor) > 1) {
            n_per_row = src_tensor->ne[0];
            nrows = nelements / n_per_row;
        }

        // Check alignment with block size
        int64_t blk_size = ggml_blck_size(dst_type);
        if (n_per_row % blk_size != 0) {
            fprintf(stderr, "Warning: tensor %s has n_per_row=%lld not divisible by "
                    "block_size=%lld for %s, keeping original type\n",
                    name, (long long)n_per_row, (long long)blk_size,
                    ggml_type_name(dst_type));
            gguf_add_tensor(dst_gctx, src_tensor);
            total_dst_bytes += src_nbytes;
            n_kept++;
            continue;
        }

        // Allocate output buffer (persists until write)
        size_t dst_nbytes = ggml_row_size(dst_type, n_per_row) * nrows;
        quant_buffers.emplace_back(dst_nbytes);
        std::vector<uint8_t> & dst_data = quant_buffers.back();

        // Initialize quantization tables
        ggml_quantize_init(dst_type);

        // Quantize
        size_t actual_bytes = ggml_quantize_chunk(
            dst_type,
            f32_data.data(),
            dst_data.data(),
            0,          // start row
            nrows,
            n_per_row,
            nullptr     // no importance matrix
        );

        if (actual_bytes != dst_nbytes) {
            // Resize to actual
            dst_data.resize(actual_bytes);
            dst_nbytes = actual_bytes;
        }

        // Create tensor metadata for output
        struct ggml_init_params tmp_params = {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * tmp_ctx = ggml_init(tmp_params);
        tmp_contexts.push_back(tmp_ctx);

        struct ggml_tensor * dst_tensor = ggml_new_tensor(
            tmp_ctx, dst_type, ggml_n_dims(src_tensor), src_tensor->ne);
        ggml_set_name(dst_tensor, name);
        dst_tensor->data = dst_data.data();

        gguf_add_tensor(dst_gctx, dst_tensor);
        gguf_set_tensor_data(dst_gctx, name, dst_data.data());

        total_dst_bytes += dst_nbytes;
        n_quantized++;

        float ratio = (float)src_nbytes / dst_nbytes;
        fprintf(stderr, "  [%3d/%d] %-50s  %6s → %6s  %.1fx (%s)\n",
                i + 1, n_tensors, name,
                ggml_type_name(src_type), ggml_type_name(dst_type),
                ratio, tc_name(tc));
    }

    // ── Write output GGUF ─────────────────────────────────────
    fprintf(stderr, "\nWriting %s...\n", output_path);
    if (!gguf_write_to_file(dst_gctx, output_path, false)) {
        fprintf(stderr, "Failed to write output GGUF\n");
        return 1;
    }

    float compression = (float)total_src_bytes / total_dst_bytes;
    fprintf(stderr, "\nDone!\n");
    fprintf(stderr, "  Tensors: %d quantized, %d kept\n", n_quantized, n_kept);
    fprintf(stderr, "  Size: %.2f GB → %.2f GB (%.1fx compression)\n",
            total_src_bytes / 1e9, total_dst_bytes / 1e9, compression);

    // Cleanup
    ggml_quantize_free();
    gguf_free(dst_gctx);
    gguf_free(src_gctx);
    ggml_free(src_ctx);
    for (auto * ctx : tmp_contexts) {
        ggml_free(ctx);
    }

    return 0;
}
