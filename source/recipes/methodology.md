# How deployments are benchmarked

A recipe is a deployment configuration for one Hub model on one Amazon SageMaker AI instance type. The catalog lists a recipe only after it was deployed on a SageMaker real-time endpoint with the Hugging Face vLLM Deep Learning Container (DLC) and passed at least the chat check described below. A recipe pins everything that changes the result:

- **Model**: the Hub model ID and the exact model revision, passed to the container as `SM_VLLM_REVISION`.
- **Container**: the Hugging Face vLLM DLC `huggingface-vllm`, pinned by image digest when available, which is the case for nearly all recipes, and by tag otherwise.
- **Instance**: the SageMaker instance type, for example `ml.g6e.xlarge` with one NVIDIA L40S GPU.
- **Engine settings**: the `SM_VLLM_*` environment variables passed to the container. Each one maps to a vLLM engine argument, so `SM_VLLM_TENSOR_PARALLEL_SIZE=4` becomes `--tensor-parallel-size 4` and `SM_VLLM_MAX_MODEL_LEN` sets the context length.
- **Endpoint settings**: the inference AMI version, for example `al2023-ami-sagemaker-inference-gpu-4-1`, and the container startup health check timeout.

Each model page shows these settings, the date and AWS region of the validation run, and the vLLM version. The AWS CloudFormation template behind the deploy button and the AWS CLI command is built from the same settings.

## Validation

Validation sends a few requests to the endpoint, one per capability, and checks each response:

- **Chat**: a chat completion request returns a well-formed answer.
- **Streaming**: the same request with `stream: true` returns the answer as server-sent events.
- **Tool calling**: for a request that defines a tool, the model returns a structured tool call.
- **Reasoning**: with reasoning turned on, the reasoning parser returns the reasoning separately from the final answer. A second request turns reasoning off and expects an answer without reasoning.
- **Image and video input**: the model describes an image or a short video, for models that accept them.

Each check only runs for capabilities the model is expected to support, so a text-only model is not tested on images. Chat must pass for a recipe to be listed. The other checks are informative: a model page shows the **Tools**, **Reasoning**, and **Vision** tags only when the corresponding check passed, and lists the checks that failed below the instance. A failed check means the capability did not work through SageMaker with these settings, so don't rely on it for that recipe.

Validation also sends a request to the OpenAI Responses API (`/v1/responses`). The SageMaker OpenAI-compatible route returned HTTP 404 for these requests in our runs, so model pages don't report this check. Use the chat completions API, as shown in [Deploy a model](./deploy-a-recipe).

## Benchmarks

Benchmarks use [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve.html) against the OpenAI-compatible route of the SageMaker endpoint, so the numbers include the SageMaker front end and the network between the benchmark client and the AWS region, not only the engine. Some recipes have no benchmark results yet, and their model page says so.

Each workload models a different use. Input and output lengths are nominal: the random prompts vary in length, and the measured averages are close to these values.

| Workload on the model page | Concurrent requests | Input tokens | Output tokens | Use |
|---|---:|---:|---:|---|
| One user (`single-request`) | 1 | about 1,024 | about 2,048 | One user at a time; shows the best speed a single user sees |
| Long answers (`long-decode`) | 4 | about 1,024 | about 4,096 | Long answers, such as reports or code |
| Long prompts (`long-prefill`) | 4 | about 16,384 | about 128 | Long prompts with short answers, such as retrieval or questions about a document |
| 64 users (`concurrency`) | 64 | about 1,024 | about 256 | Many users at once; shows the throughput of the instance |
| Shared prompt prefix (`prefix`) | 32 | about 4,096 shared + 256 | about 128 | Requests that share a long prefix, such as a system prompt; shows the effect of prefix caching |
| Real conversations (`sharegpt`) | 256 | varies | varies | 1,000 real conversations from the ShareGPT dataset |

No workload uses more than 256 concurrent requests. Higher concurrency produced HTTP 429 responses in our runs, which would measure the rejected requests instead of the model.

## How to read the metrics

- **Tokens/s per user**: how fast one user sees tokens arrive, computed as 1,000 divided by the median time per output token (TPOT) in milliseconds. Higher is better. The One user workload gives the best case.
- **Time to first token**: the median delay between sending a request and receiving the first token (TTFT). It grows with the prompt length and with load. Lower is better.
- **Total tokens/s**: output tokens per second summed over all concurrent requests. It measures the capacity of the instance. Higher is better.
- **Cost per 1M output tokens**: the hourly price of the instance divided by the output tokens generated in one hour at the measured total tokens/s. It assumes the endpoint is busy all the time, so a real workload with idle periods costs more per token.

Prices are SageMaker AI on-demand prices in `us-east-1` at the time the catalog was generated, whatever the region of the validation run. Prices differ in other regions and with savings plans; check [Amazon SageMaker AI pricing](https://aws.amazon.com/sagemaker/ai/pricing/) for yours. `n/a` means the value was not measured.
