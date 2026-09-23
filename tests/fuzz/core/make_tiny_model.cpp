// make_tiny_model: write a tiny Qwen3.5 GGUF with random weights for the decode harnesses.
//
//   make_tiny_model <out.gguf> [f32|q8_0|q4_0] [seed]
//
// The model has the architecture qwen35 of llama.cpp: 4 layers, where layers
// 0, 1 and 2 are gated delta net (recurrent) layers and layer 3 is a full
// attention layer (full_attention_interval = 4, as in the real model), with
// n_embd 256, 2 heads of 128, n_ff 384, a state of 128 and a vocabulary of 512
// ids without a tokenizer. The head and state sizes are the sizes of the real
// model, thus the Hexagon kernels take the same paths.
//
// The program builds the metadata with llama_model_saver (the method of
// tests/test-llama-archs.cpp), creates the model with
// llama_model_init_from_user, fills each tensor with N(0, 0.02) from a seeded
// generator, and saves an F32 file. For q8_0 or q4_0 it then quantizes the
// file with llama_model_quantize and removes the F32 file.

#include "llama.h"
#include "gguf.h"

#include "llama-arch.h"
#include "llama-model-saver.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kVocab    = 512;
constexpr uint32_t kCtx      = 4096;
constexpr uint32_t kEmbd     = 256;
constexpr uint32_t kHead     = 2;
constexpr uint32_t kFf       = 384;
constexpr uint32_t kLayer    = 4;

/**
 * The standard deviation of the weights of one tensor. A matrix gets
 * 1 / sqrt(fan-in), thus each layer changes the residual stream by an amount
 * of order 1, and the logits depend on the history. With small weights the
 * logits of the last token do not depend on the earlier tokens, and the
 * oracle of fuzz_recurrent cannot see a wrong state.
 */
float weight_std(const ggml_tensor * tensor) {
    // A small embedding: the layer outputs then dominate the residual stream, and the
    // logits of the last token depend on the earlier tokens. The norm of each layer
    // input gives the layers an input of order 1 all the same.
    if (strstr(tensor->name, "token_embd") != nullptr) {
        return 0.05f;
    }
    if (tensor->ne[1] > 1) {
        return 1.0f / std::sqrt((float) tensor->ne[0]);
    }
    return 0.1f;
}

/** Fill one tensor from a generator seeded by the seed and the tensor name. */
void set_tensor_data(ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    seed ^= std::hash<std::string>{}(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, weight_std(tensor));

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (auto & v : tmp) {
            v = dis(gen);
        }
        if (strstr(tensor->name, "norm") != nullptr) {
            // the norm weights multiply the activations: a value near 1 keeps the scale of the model
            for (auto & v : tmp) {
                v = 1.0f + 0.2f * v;
            }
        } else if (strstr(tensor->name, "ssm_a") != nullptr) {
            // A = -exp(A_log) < 0 gives a decay below 1, thus the state does not grow without limit
            for (auto & v : tmp) {
                v = -(0.05f + std::fabs(v));
            }
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (auto & v : tmp) {
            v = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        fprintf(stderr, "make_tiny_model: tensor %s has the type %s, which this program cannot fill\n",
                tensor->name, ggml_type_name(tensor->type));
        exit(1);
    }
}

/** The metadata of the tiny qwen35 model. */
gguf_context * tiny_metadata() {
    gguf_context * ctx = gguf_init_empty();
    llama_model_saver ms(LLM_ARCH_QWEN35, ctx);
    const uint32_t n_embd_head = kEmbd / kHead;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(LLM_ARCH_QWEN35));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  kVocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              kCtx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            kEmbd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 kLayer);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         kFf);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        kHead);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     kHead);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-6f);
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE,              10000000.0f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,     std::vector<uint32_t>({ n_embd_head / 4, n_embd_head / 4, n_embd_head / 4, n_embd_head / 4 }));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,     uint32_t(4));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,              uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,             uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,              uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,          uint32_t(2));
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,             uint32_t(2));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");
    return ctx;
}

/** A silent progress callback. */
bool silent(float /*progress*/, void * /*user_data*/) {
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out.gguf> [f32|q8_0|q4_0] [seed]\n", argv[0]);
        return 2;
    }
    const std::string out   = argv[1];
    const std::string type  = argc > 2 ? argv[2] : "f32";
    size_t            seed  = argc > 3 ? strtoull(argv[3], nullptr, 0) : 20260923;

    llama_backend_init();
    gguf_context * meta = tiny_metadata();

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent;
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[] = { cpu, nullptr };
    mp.devices = devs;

    llama_model * model = llama_model_init_from_user(meta, set_tensor_data, &seed, mp);
    if (model == nullptr) {
        fprintf(stderr, "make_tiny_model: llama_model_init_from_user failed. Run with a log to see the missing key or tensor.\n");
        return 1;
    }

    const std::string f32_path = type == "f32" ? out : out + ".f32.tmp";
    llama_model_save_to_file(model, f32_path.c_str());
    llama_model_free(model);
    gguf_free(meta);

    if (type != "f32") {
        llama_model_quantize_params qp = llama_model_quantize_default_params();
        if (type == "q8_0") {
            qp.ftype = LLAMA_FTYPE_MOSTLY_Q8_0;
        } else if (type == "q4_0") {
            qp.ftype = LLAMA_FTYPE_MOSTLY_Q4_0;
        } else {
            fprintf(stderr, "make_tiny_model: the type %s is not known. Use f32, q8_0 or q4_0.\n", type.c_str());
            remove(f32_path.c_str());
            return 2;
        }
        qp.nthread = 4;
        const uint32_t rc = llama_model_quantize(f32_path.c_str(), out.c_str(), &qp);
        remove(f32_path.c_str());
        if (rc != 0) {
            fprintf(stderr, "make_tiny_model: llama_model_quantize failed with the code %u\n", rc);
            return 1;
        }
    }
    printf("make_tiny_model: wrote %s (%s, seed %zu)\n", out.c_str(), type.c_str(), seed);
    return 0;
}
