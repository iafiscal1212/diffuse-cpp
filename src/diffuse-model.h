#pragma once

#include "diffuse-common.h"

// Load model weights from GGUF file
diffuse_model * diffuse_model_load_impl(const std::string & path, int n_threads);

// Free model and all associated memory
void diffuse_model_free_impl(diffuse_model * model);
