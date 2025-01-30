# 1.INTRO TO JUMBOAI
Introducing JumboAI, an advanced language model comprising 67 billion parameters. It has been trained from scratch on a vast dataset of 2 trillion tokens in both English and Chinese. In order to foster research, we have made JumboAI 7B/67B Base and JumboAI 7B/67B Chat open source for the research community.
![image](https://github.com/user-attachments/assets/d20657e8-3450-4df9-91d3-b7cafedc42e9)

-Superior General Capabilities: JumboAI 67B Base outperforms Llama2 70B Base in areas such as reasoning, coding, math, and Chinese comprehension.

-Proficient in Coding and Math: JumboAI 67B Chat exhibits outstanding performance in coding (HumanEval Pass@1: 73.78) and mathematics (GSM8K 0-shot: 84.1, Math 0-shot: 32.6). It also demonstrates remarkable generalization abilities, as evidenced by its exceptional score of 65 on the Hungarian National High School Exam.

-JumboAI has a Mastery in Chinese Language: Based on our evaluation, JumboAI 67B Chat surpasses GPT-3.5 in Chinese.


# 2. EVALUATION RESULTS

Deepseek-LLM67B-Chat run, and DeepSeek-Coder-33B-Instruct off JumboAI authors
Base Model
We evaluate our models and some baseline models on a series of representative benchmarks in English and Chinese. More results can be found in the evaluation folder. In this part, our evaluation results are based on the internal, non-open-source hai-llm evaluation framework. Please note that there may be slight discrepancies when using the converted HuggingFace models.

![Screenshot 2025-01-30 083833](https://github.com/user-attachments/assets/f9fe4081-5db7-4b1f-8c0f-1977f12058bc)

Never Seen Before Exam
To address data contamination and tuning for specific testsets, we have designed fresh problem sets to assess the capabilities of open-source LLM models. The evaluation results indicate that DeepSeek LLM 67B Chat performs exceptionally well on never-before-seen exams.

Hungarian National High-School Exam: In line with Grok-1, we have evaluated the model's mathematical capabilities using the Hungarian National High School Exam. This exam comprises 33 problems, and the model's scores are determined through human annotation. We follow the scoring metric in the solution.pdf to evaluate all models.

![image](https://github.com/user-attachments/assets/1ff3d324-fe74-449a-9f81-bd05ae1c4a59)

Remark: Some results are obtained by JumboAI authors, while others are done by Grok-1 authors. We found some models count the score of the last question (Llemma 34b and Mammoth) while some (MetaMath-7B) are not in the original evaluation. In our evaluation, we count the last question score. Evaluation details will be made public shortly.

Instruction Following Evaluation: On Nov 15th, 2023, Google released an instruction following evaluation dataset. They identified 25 types of verifiable instructions and constructed around 500 prompts, with each prompt containing one or more verifiable instructions. We use the prompt-level loose metric to evaluate all models.

![image](https://github.com/user-attachments/assets/a6d0bedf-8555-4192-be59-1c1b796f0d43)

LeetCode Weekly Contest: To assess the coding proficiency of the model, we have utilized problems from the LeetCode Weekly Contest (Weekly Contest 351-372, Bi-Weekly Contest 108-117, from July 2023 to Nov 2023). We have obtained these problems by crawling data from LeetCode, which consists of 126 problems with over 20 test cases for each. The evaluation metric employed is akin to that of HumanEval. In this regard, if a model's outputs successfully pass all test cases, the model is considered to have effectively solved the problem. The model's coding capabilities are depicted in the Figure below, where the y-axis represents the pass@1 score on in-domain human evaluation testing, and the x-axis represents the pass@1 score on out-domain LeetCode Weekly Contest problems.

![image](https://github.com/user-attachments/assets/6b07f907-2331-427d-85a5-d57278acacfe)

The specific questions and test cases will be released soon. Stay tuned!

![Screenshot 2025-01-30 084405](https://github.com/user-attachments/assets/4953e641-9eb0-4850-ab6b-463b41f47064)

Note: We evaluate chat models with 0-shot for MMLU, GSM8K, C-Eval, and CMMLU. More evaluation results can be found soon!

Revisit Multi-Choice Question Benchmarks

Based on our experimental observations, we have discovered that enhancing benchmark performance using multi-choice (MC) questions, such as MMLU, CMMLU, and C-Eval, is a relatively straightforward task. By incorporating multi-choice questions from Chinese exams, we have achieved exceptional results, as depicted in the table below:

![Screenshot 2025-01-30 084535](https://github.com/user-attachments/assets/a2e4b8b3-eb7d-4a55-82dd-0437f1c44281)

Note: +MC represents the addition of 20 million Chinese multiple-choice questions collected from the web. It is important to note that we conducted deduplication for the C-Eval validation set and CMMLU test set to prevent data contamination. This addition not only improves Chinese multiple-choice benchmarks but also enhances English benchmarks. However, we observed that it does not enhance the model's knowledge performance on other evaluations that do not utilize the multiple-choice style in the 7B setting. As a result, we made the decision to not incorporate MC data in the pre-training or fine-tuning process, as it would lead to overfitting on benchmarks.

# 3. Pre-Training Details

# Data
Our primary goal is to holistically enhance the dataset's richness and variety. To achieve this, we've implemented multiple methods and established a distributed, frequent-checkpointing batch processing system, named "cc_cleaner", to bolster our data pipeline.

Our minimal viable solution departs from RefinedWeb + CCNet. We greatly appreciate their selfless dedication to the research of AGI.

We have also significantly incorporated deterministic randomization into our data pipeline. This approach enables us to continuously enhance our data throughout the lengthy and unpredictable training process.

    -Data Composition: Our training data comprises a diverse mix of Internet text, math, code, books, and self-collected data respecting robots.txt. In addition to the diverse content, we place a high priority on personal privacy and copyright protection. All content containing personal information or subject to 
    copyright restrictions has been removed from our dataset.

    -Dataset Pruning: Our system employs heuristic rules and models to refine our training data. Our filtering process removes low-quality web data while preserving precious low-resource knowledge. It aims to improve overall corpus quality and remove harmful or toxic content.
    
    -Deduplication: Our advanced deduplication system, using MinhashLSH, strictly removes duplicates both at document and string levels. This rigorous deduplication process ensures exceptional data uniqueness and integrity, especially crucial in large-scale datasets.

# Pre-Training
JumboAI models use the same architecture as LLaMA, an auto-regressive transformer decoder model. The 7B model uses Multi-Head attention (MHA) while the 67B model uses Grouped-Query Attention (GQA).

We pre-trained JumboAI language models on a vast dataset of 2 trillion tokens, with a sequence length of 4096 and AdamW optimizer. The 7B model's training involved a batch size of 2304 and a learning rate of 4.2e-4 and the 67B model was trained with a batch size of 4608 and a learning rate of 3.2e-4. We employ a multi-step learning rate schedule in our training process. The learning rate begins with 2000 warmup steps, and then it is stepped to 31.6% of the maximum at 1.6 trillion tokens and 10% of the maximum at 1.8 trillion tokens.

We release the training loss curve and several benchmark metrics curves, as detailed below.

![image](https://github.com/user-attachments/assets/f8e8651e-33d8-468a-a42e-f89d3e75d55e)
![image](https://github.com/user-attachments/assets/327463cd-d59b-49e5-8a2a-7c9225c2df5d)


# 4. Quick Start

###Installation
On the basis of Python >= 3.8 environment, install the necessary dependencies by running the following command:
```shell
pip install -r requirements.txt
```

### Inference

Here are some examples of utilizing our models.

**Text Completion**

You can directly employ [Huggingface's Transformers](https://github.com/huggingface/transformers) for model inference:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "JumboAI-ai/JumboAI-llm-67b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

**Chat Completion**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "JumboAI-ai/JumboAI-llm-67b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

messages = [
    {"role": "user", "content": "Who are you?"}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)
```

Avoiding the use of the provided function `apply_chat_template`, you can also interact with our model following the sample template. Note that `messages` should be replaced by your input.

```
User: {messages[0]['content']}

Assistant: {messages[1]['content']}<｜end▁of▁sentence｜>User: {messages[2]['content']}

Assistant:
```

**Note:** By default (`add_special_tokens=True`), our tokenizer automatically adds a `bos_token` (`<｜begin▁of▁sentence｜>`) before the input text. Additionally, since the system prompt is not compatible with this version of our models, we DO NOT RECOMMEND including the system prompt in your input.

**Inference with vLLM**

You can also employ [vLLM](https://github.com/vllm-project/vllm) for high-throughput inference.

```python
from vllm import LLM, SamplingParams

tp_size = 4 # Tensor Parallelism
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
model_name = "JumboAI-ai/JumboAI-llm-67b-base"
llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=tp_size)

prompts = [
    "If everyone in a country loves one another,",
    "The research should also focus on the technologies",
    "To determine if the label is correct, we need to"
]
outputs = llm.generate(prompts, sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
```

## 5. FAQ

### Could You Provide the tokenizer.model File for Model Quantization?

JumboAI LLM utilizes the [HuggingFace Tokenizer](https://huggingface.co/docs/tokenizers/index) to implement the Byte-level BPE algorithm, with specially designed pre-tokenizers to ensure optimal performance. Currently, there is no direct way to convert the tokenizer into a SentencePiece tokenizer. We are contributing to the open-source quantization methods facilitate the usage of HuggingFace Tokenizer.

#### GGUF(llama.cpp)

We have submitted a [PR](https://github.com/ggerganov/llama.cpp/pull/4070) to the popular quantization repository [llama.cpp](https://github.com/ggerganov/llama.cpp) to fully support all HuggingFace pre-tokenizers, including ours.

While waiting for the PR to be merged, you can generate your GGUF model using the following steps:

```bash
git clone https://github.com/DOGEwbx/llama.cpp.git
cd llama.cpp
git checkout regex_gpt2_preprocess
# set up the environment according to README
make
python3 -m pip install -r requirements.txt
# generate GGUF model
python convert-hf-to-gguf.py <MODEL_PATH> --outfile <GGUF_PATH> --model-name deepseekllm
# use q4_0 quantization as an example
./quantize <GGUF_PATH> <OUTPUT_PATH> q4_0
./main -m <OUTPUT_PATH> -n 128 -p <PROMPT>
```
#### GPTQ(exllamav2)

`UPDATE:`[exllamav2](https://github.com/turboderp/exllamav2) has been able to support HuggingFace Tokenizer. Please pull the latest version and try out.

### GPU Memory Usage

We profile the peak memory usage of inference for 7B and 67B models at different batch size and sequence length settings.

For JumboAI LLM 7B, we utilize **1 NVIDIA A100-PCIE-40GB GPU** for inference.

<table><thead><tr><th rowspan="2">Batch Size</th><th colspan="5">Sequence Length</th></tr><tr><th>256</th><th>512</th><th>1024</th><th>2048</th><th>4096</th></tr></thead><tbody><tr><td>1</td><td>13.29 GB</td><td>13.63 GB</td><td>14.47 GB</td><td>16.37 GB</td><td>21.25 GB</td></tr><tr><td>2</td><td>13.63 GB</td><td>14.39 GB</td><td>15.98 GB</td><td>19.82 GB</td><td>29.59 GB</td></tr><tr><td>4</td><td>14.47 GB</td><td>15.82 GB</td><td>19.04 GB</td><td>26.65 GB</td><td>OOM</td></tr><tr><td>8</td><td>15.99 GB</td><td>18.71 GB</td><td>25.14 GB</td><td>35.19 GB</td><td>OOM</td></tr><tr><td>16</td><td>19.06 GB</td><td>24.52 GB</td><td>37.28 GB</td><td>OOM</td><td>OOM</td></tr></tbody></table>

For JumboAI LLM 67B, we utilize **8 NVIDIA A100-PCIE-40GB GPUs** for inference.

<table><thead><tr><th rowspan="2">Batch Size</th><th colspan="5">Sequence Length</th></tr><tr><th>256</th><th>512</th><th>1024</th><th>2048</th><th>4096</th></tr></thead><tbody><tr><td>1</td><td>16.92 GB</td><td>17.11 GB</td><td>17.66 GB</td><td>20.01 GB</td><td>33.23 GB</td></tr><tr><td>2</td><td>17.04 GB</td><td>17.28 GB</td><td>18.55 GB</td><td>25.27 GB</td><td>OOM</td></tr><tr><td>4</td><td>17.20 GB</td><td>17.80 GB</td><td>21.28 GB</td><td>33.71 GB</td><td>OOM</td></tr><tr><td>8</td><td>17.59 GB</td><td>19.25 GB</td><td>25.69 GB</td><td>OOM</td><td>OOM</td></tr><tr><td>16</td><td>18.17 GB</td><td>21.69 GB</td><td>34.54 GB</td><td>OOM</td><td>OOM</td></tr></tbody></table>

## 6. Limitation

While JumboAI LLMs have demonstrated impressive capabilities, they are not without their limitations. Here are some potential drawbacks of such models:

1. Over-reliance on training data: These models are trained on vast amounts of text data, which can introduce biases present in the data. They may inadvertently generate biased or discriminatory responses, reflecting the biases prevalent in the training data.

2. Hallucination: The model sometimes generates responses or outputs that may sound plausible but are factually incorrect or unsupported. This can occur when the model relies heavily on the statistical patterns it has learned from the training data, even if those patterns do not align with real-world knowledge or facts.

3. Repetition: The model may exhibit repetition in their generated responses. This repetition can manifest in various ways, such as repeating certain phrases or sentences, generating redundant information, or producing repetitive structures in the generated text. This issue can make the output of LLMs less diverse and less engaging for users.

## 7. License

JumboAI LLM series (including Base and Chat) supports commercial use.

## 8. Citation

```
@misc{JumboAI-llm,
  author = {JumboAI AI},
  title = {JumboAI LLM: Let there be answers},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JumboAI-ai/JumboAI-LLM}},
}
```

## 9. Contact

If you have any questions, please raise an issue or contact us at JumboAI.xyz.
