#include "common.h"
#include "llama.h"

#include <ctime>
#include <iostream>
#include <string>
#include <sstream>


/**
 * protocol:
 * - n_embd * n_tokens matrix of embeddings
 * - tokens of n_ctx generated so far
 *
 * This state should be accumulated in context of each calls, so there is no pollution of data in the protocol.
 * It also make this thing scale with the size of the context, as long as it fits in memory
 */

int main(int argc, char ** argv) {
    int first_layer = std::stoi(argv[1]);
    int last_layer = std::stoi(argv[2]);


    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";

    if (gpt_params_parse(argc-2, argv+2, params) == false) {
        return 1;
    }

    params.embedding = true;

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                        "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_context * ctx;
    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.logits_all = params.perplexity;
        lparams.use_mmap   = params.use_mmap;
        lparams.use_mlock  = params.use_mlock;
        lparams.embedding  = params.embedding;
        lparams.part       = {first_layer, last_layer};

        ctx = llama_init_from_file(params.model.c_str(), lparams);
        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    int n_past = 0;

    const int n_embd = llama_n_embd(ctx);
    const int n_layer = llama_n_layer(ctx);

    std::vector<llama_token> embd_inp;
    std::vector<float> embd;
    bool has_tokens_as_input = first_layer == 0;
    bool has_tokens_as_output = last_layer == n_layer;

    if (has_tokens_as_input) {
        // Add a space in front of the first character to match OG llama tokenizer behavior
        params.prompt.insert(0, 1, ' ');

        // tokenize the prompt
        embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    } else {
        for (std::string line; std::getline(std::cin, line);) {
            std::string segment;
            std::stringstream ss(line);
            while(std::getline(ss, segment, ' ')) {
                embd.push_back(std::stof(segment));
            }
        }
    }


    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
        }
        fprintf(stderr, "\n");
    }

    {
        int ret;
        if (has_tokens_as_input) {
            ret = llama_eval(
                    ctx,
                    {embd_inp.data(), NULL, (int) embd_inp.size(), 0},
                    n_past,
                    params.n_threads);
        } else {
            ret = llama_eval(
                    ctx,
                    {NULL, embd.data(), (int) embd.size() / n_embd, n_embd},
                    n_past,
                    params.n_threads);
        }
        if (ret) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }
    }

    if (has_tokens_as_output) {
        const int32_t top_k          = params.top_k;
        const float   top_p          = params.top_p;
        const float   temp           = params.temp;
        const float   repeat_penalty = params.repeat_penalty;
        // const int n_ctx = llama_n_ctx(ctx);

        auto last_n_tokens = std::vector<llama_token>(params.repeat_last_n, 0);

        llama_token id = llama_sample_top_p_top_k(ctx,
                                      &last_n_tokens.back() - params.repeat_last_n,
                                      params.repeat_last_n, top_k, top_p, temp, repeat_penalty);
        printf("%s\n", llama_token_to_str(ctx, id));
    } else {
        const auto embeddings = llama_get_embeddings(ctx);

        for (int j = 0; j < embd_inp.size(); j++) {
            for (int i = 0; i < n_embd; i++) {
                printf("%f", embeddings[j * llama_n_embd(ctx) + i]);
                if (i < n_embd - 1) {
                    printf(" ");
                }
            }
            printf("\n");
        }
    }

    llama_print_timings(ctx);
    llama_free(ctx);

    return 0;
}
