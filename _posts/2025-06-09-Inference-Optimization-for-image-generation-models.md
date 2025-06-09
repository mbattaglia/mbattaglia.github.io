---
layout: post
title:  "Optimizing Inference for Image Generation Models: Memory Tricks and Quantization"
author: Matías Battaglia
excerpt: |
  <div class="excerpt-container">
    <div class="excerpt-image">
      <img src="/assets/images/posts/2025-06-09-image-generation.gif" alt="Model quantization">
    </div>
    <div class="excerpt-text">
    Let's explore how I was able to run an image generation model, FLUX.1 Dev, with only 20% of its required total VRAM. Through quantization and memory optimization techniques, I'll show you practical strategies that make high-quality image generation accessible on consumer GPUs, complete with performance benchmarks and real-world examples.
    </div>
  </div>
---
Over the past few months, I've been exploring how to optimize inference for image generation models. The motivation was simple: I wanted to generate images locally on my desktop GPU, but the 35GB VRAM requirement to fully load the model meant it wasn't going to happen.

In this post, I'll discuss some inference tricks and quantization techniques, courtesy of the amazing work done by Hugging Face in the Diffusers and bitsandbytes libraries.

> I will be covering some of this topics in an upcoming conference talk in the AWS Summit Madrid 2025. The session ID is **AIM306**. Hope to see you there!

## TL;DR: Quick Start Guide

<iframe id="memory-chart" src="/assets/html/2025-06-09-memory_usage_chart.html" width="100%" style="border:none;"></iframe>
<script>
  window.addEventListener("message", function(event) {
    if (event.data && event.data.type === "setHeight") {
      var iframe = document.getElementById("memory-chart");
      if (iframe) {
        iframe.style.height = event.data.height + "px";
      }
    }
  });
</script>
(graph courtesy of Claude's interactive artifacts)

- **Best overall recommendation**: Start with **NF4 Quantization** - it cuts VRAM usage in half with minimal quality loss and similar total inference time compared to full precision.
- **Pro tip**: The `enable_model_cpu_offload()` method is your friend for mid-range GPUs. It's much faster than sequential offloading while still saving significant memory.

## A Quick Introduction to Image Generation Models

Before diving into the optimization techniques I explored, I want to spend some time explaining image generation models: what they are, how they generate images, and the unique challenges they present.

### Diffusion Models

Diffusion models have been, for the most part, the gold standard for image generation (at least until very recently). At their core, these models work by learning to reverse a noise process. Think of it like this: imagine you have a clear image that gradually gets more and more noisy until it becomes pure static. A diffusion model learns to reverse this process, starting from pure noise and gradually removing it to reveal a coherent image.

![Denoising process](/assets/images/posts/2025-06-09-image-generation.gif)

The process happens iteratively in steps, where at each step, the model predicts what noise should be removed. This approach is both a strength and a challenge. It's a strength because it allows for incredibly detailed and high-quality outputs. It's a challenge because it means the model needs to run multiple times for a single image, making it computationally expensive and potentially slower.

In this post, I'll focus primarily on [FLUX.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), an image generation model that revolutionized image generation when it [was released](https://bfl.ai/announcements/24-08-01-bfl) by Black Forest Labs in August 2024. This model departs from previous architectures (like Stable Diffusion) and instead uses transformers, the [same architecture](https://arxiv.org/abs/1706.03762) that revolutionized deep learning and kicked off the LLM era we now live in.

It's worth noting that Black Forest Labs initially released a family of models: FLUX.1 Pro, FLUX.1 Dev, and FLUX.1 Schnell. The Pro model is only available through APIs, and its weights were not released openly. Dev is open-weight, with a non-commercial applications license. Schnell was released under an Apache 2.0 license. Since then, Black Forest Labs has released more models, such as [Kontext](https://bfl.ai/announcements/flux-1-kontext).

### Transformer Models

FLUX.1 uses what's called a "diffusion transformer" (also known as a DiT architecture). The key advantage of transformers in image generation is their ability to handle complex relationships within an image. Unlike convolutional layers that focus on local features, transformers can relate any part of an image to any other part directly. This leads to more coherent compositions and better understanding of global image structure.

However, transformers come with their own computational overhead. The attention mechanism that makes them so powerful scales quadratically with the input size, making them memory-hungry and computationally intensive. This is precisely where quantization comes into play.

FLUX.1 consists of multiple components: a scheduler, two text encoders, the diffusion transformer already explained, a VAE (Variational Autoencoder), among others. Let me explain some of them:

- The text encoders process and understand your written prompts, converting natural language into numerical representations that the model can work with.
- This information is then passed to the diffusion transformer, which uses the encoded text information to iteratively refine noise into coherent images.
- Finally, the VAE handles the conversion between the compressed latent space where generation happens and the final high-resolution images you see.

## Running FLUX.1

FLUX.1 Dev's weights have been released on Hugging Face and integrated with the [diffusers library](https://github.com/huggingface/diffusers), making it extremely convenient to run the image generation pipeline pretty much anywhere.

Here's an example from [diffuser's documentation](https://huggingface.co/docs/diffusers/en/api/pipelines/flux#guidance-distilled) (with a modified prompt):

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

prompt = "A group of playful baby foxes having a picnic in a magical forest, with tiny teacups, mushroom tables, and glowing fireflies. Style: Whimsical children's book illustration."

out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]

out.save("image.png")
```

Please note that since FLUX.1 Dev is a gated model, you'll need to be logged into Hugging Face and accept its license before downloading the model.

Running this code in Google Colab generates this image after 26 seconds:

![First example](/assets/images/posts/2025-06-09-image-generation-example1.png)

The image quality is quite good, and it does an excellent job adhering to the nuanced prompt I provided. There are a couple of small artifacts that I could probably prompt away, but it's not bad at all.

This also required a whopping 39.27 GB of VRAM in total, meaning it barely fit in the largest runtime available, the A100.

Let's start discussing how to optimize this so we can run the code with less VRAM.

## Memory Usage Optimizations

Before implementing any optimization, let's define a way to check how much VRAM we're using:

```python
import torch
def print_gpu_memory():
    free_mem, total_mem = torch.cuda.mem_get_info()
    used_mem = total_mem - free_mem
    print(f"GPU memory used: {used_mem / 1e9:.2f} GB")
    print(f"GPU memory free: {free_mem / 1e9:.2f} GB")
    print(f"GPU memory total: {total_mem / 1e9:.2f} GB")

print_gpu_memory()
```

We'll also use some code to free up VRAM between invocations:

```python
import gc

def flush():
    """Clears GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
```

Let's start using some of the techniques [readily available in the Diffusers library](https://huggingface.co/docs/diffusers/en/optimization/memory#reduce-memory-usage) to optimize memory usage.

### CPU Offloading

Diffusers [implements CPU offloading](https://huggingface.co/docs/diffusers/en/optimization/memory#model-offloading), which offloads the weights to the CPU and only loads them on the GPU when performing the forward pass. This works at the submodule level, meaning it will offload and load submodels as needed, resulting in a large number of memory transfers. This has the benefit of saving memory, but the tradeoff is that inference will take much longer, especially for high numbers of diffusion iterations (controlled by `num_inference_steps`).

You can enable it by calling `enable_sequential_cpu_offload()` on the pipeline before generating an image:

```python
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.enable_sequential_cpu_offload()

prompt = "A group of playful baby foxes having a picnic in a magical forest, with tiny teacups, mushroom tables, and glowing fireflies. Style: Whimsical children's book illustration."

out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]
out.save("image.png")
```

Running this code took just over 5 minutes in the same Colab environment as the previous example and produced the exact same image. VRAM usage peaked at around 2GB, so this could be a way to run models very slowly with low VRAM requirements.

### Model Offloading

Model offloading is another option that moves entire models back and forth between CPU and GPU, instead of handling it at the submodel level. Keep in mind that not all modules are offloaded to the GPU, meaning that overall VRAM usage will be higher than sequential CPU offloading.

You can enable it by calling `enable_model_cpu_offload()` on the pipeline before generating an image:

```python
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.enable_model_cpu_offload()

prompt = "A group of playful baby foxes having a picnic in a magical forest, with tiny teacups, mushroom tables, and glowing fireflies. Style: Whimsical children's book illustration."

out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]
out.save("image.png")
```

Running this code took just over 56 seconds (much faster than CPU offloading) and again produced the exact same image. VRAM usage peaked at around 24GB,significantly lower than the initial 39.27 GB, but still not a small amount of memory.

This technique appears to offer an excellent tradeoff between memory usage and inference speed.

### Manual Submodel Loading and Unloading

Another approach to partially loading models into VRAM is to take advantage of the modularization of diffusers' pipelines. You could, for instance, only load the submodels when you need them and unload them afterward, significantly decreasing the total amount of maximum VRAM needed at the expense of extra latency. This approach would also allow you to use different levels of quantization for each submodel, though we'll dive deeper into that later.

Let's see an example:

```python
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import FluxPipeline
import torch

text_encoder = CLIPTextModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder",
    torch_dtype=torch.float16
)

text_encoder_2 = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder_2",
    torch_dtype=torch.float16
)

# Build a pipeline with only the text encoders
encoder_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    transformer=None,
    vae=None,
    torch_dtype=torch.float16,
).to("cuda")

print_gpu_memory()
```

This loads only the encoders in VRAM and makes them available for encoding prompts:

```python
prompt = "A group of playful baby foxes having a picnic in a magical forest, with tiny teacups, mushroom tables, and glowing fireflies. Style: Whimsical children's book illustration."

with torch.no_grad():
    print("Encoding prompts.")
    prompt_embeds, pooled_prompt_embeds, text_ids = encoder_pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,  # if not defined, "prompt" will be sent to all encoders
        max_sequence_length=256
    )
```

Memory usage at this point is **12.43 GB**.

We can then define a new pipeline for the diffusion transformer and send it the previously encoded prompts. First, we need to free up memory from the encoder and its pipeline:

```python
if 'encoder_pipe' in locals() and encoder_pipe:
    del encoder_pipe
if 'text_encoder_2' in locals() and text_encoder_2:
    del text_encoder_2
flush()
print_gpu_memory()
```

Now we can move ahead with the transformer:

```python
from diffusers import FluxTransformer2DModel

transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.float16
)

# Build a pipeline only with the transformer
transformer_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    transformer=transformer,
    torch_dtype=torch.float16,
).to("cuda")
```

Memory usage at this point: **24.54 GB**

We can now generate an image using the recently loaded diffusion transformer:

```python
out = transformer_pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]
out.save("image.png")
```

The image generation took 25 seconds, and memory peaked at **29.68 GB**.

This time, the pipeline generated a different image:

![Image generated](/assets/images/posts/2025-06-09-image-generation-example2.png)

The image quality is decent. There's definitely something going on with the foxes' eyes, but I wouldn't say it's a bad image by any means. It also added an unusual signature to the image in the bottom right.

> I believe this has to do with the encoders. I haven't been able to identify exactly what's causing this issue. I'll continue researching and will provide an update if I figure out what's happening.

#### What This Approach Enables

The architect in me loves this approach as it opens the door to decoupling the different submodels from each other. If you think about it, you could use this approach to have each submodel use different compute resources independently. This brings many advantages, including:

- Having a processing queue to optimize resource usage
- Scaling up and down based on demand (potentially to zero)
- Independent failure management for each component

## Quantization

Another way to reduce the VRAM usage of image generation models is to apply quantization. Quantization is the process of reducing the precision of numbers used in neural network computations. In simple terms, it means using fewer bits to represent the same information. Most neural networks are trained using 32-bit floating-point numbers (FP32), but quantization allows us to use 16-bit (FP16), 8-bit (INT8), or even 4-bit representations.

Think of it like the difference between a high-resolution photo and a compressed JPEG. The compressed version takes up less space and loads faster, but you lose some detail. The art of quantization is finding the sweet spot where you get significant size and speed benefits without losing too much quality.

Changing the data type means there will be a loss in precision, which means you'll get different images compared to full precision. We'll see some examples here, but I highly recommend [this blog post from Hugging Face](https://huggingface.co/blog/diffusers-quantization) that compares results using different quantization techniques.

### Types of Quantization

There are several approaches to quantization, each with different trade-offs:

- **Post-training quantization (PTQ)**: This is the simplest approach. You take a pre-trained model and convert it to lower precision without any additional training. It's fast and easy but can sometimes result in quality loss.
- **Quantization-aware training (QAT)**: This involves training the model while simulating the effects of quantization. It typically produces better results than PTQ but requires access to training data and significant computational resources.
- **Dynamic quantization**: The precision is determined at runtime based on the actual values being processed. This can provide a good balance between quality and efficiency.
- **Static quantization**: The quantization parameters are fixed ahead of time. This is more efficient at runtime but requires careful calibration.

I won't go into the details about these techniques, but I highly recommend checking out these courses by Hugging Face on DeepLearning.AI:
- [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
- [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

### How to Apply Quantization to FLUX.1

While there are multiple quantization techniques you can apply to FLUX.1 (bitsandbytes, quanto, torchao, etc.), we'll focus on bitsandbytes today. The reason is that, in my experience, it gives you the best results in terms of memory usage and inference speed. I'm happy to see that the Hugging Face team [agrees](https://huggingface.co/blog/diffusers-quantization#conclusion) in their excellent diffusers quantization blog post.

#### Quantization with bitsandbytes

In this example, we'll apply NF4 quantization at the submodel level. NF4 is a type of quantization that represents FP32 weights with just 4 bits. It's explained in the [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) paper and is ubiquitously used in quantization for image generation models.

There's a feature just released in diffusers [0.34.1](https://github.com/huggingface/diffusers/pull/11604) that would make this much easier, as it allows you to define quantization [at the pipeline level](https://github.com/huggingface/diffusers/blob/main/docs/source/en/quantization/overview.md#simple-quantization).

Since it's not currently available in PyPI as of today, and I don't want to install diffusers from source (to make this as easy as possible for everyone), I won't be using it. Here's how this looks:

```python
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import FluxPipeline, FluxTransformer2DModel
import torch

# Define quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Define text_encoder_2 in NF4
text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=torch.float16
)

# Define transformer in NF4
transformer_4bit = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16
)

# Build a pipeline with both quantized components
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder_2=text_encoder_2_4bit,
    transformer=transformer_4bit,
    torch_dtype=torch.float16
).to("cuda")

print_gpu_memory()
```

We can now run the pipeline and generate an image:

```python
prompt = "A group of playful baby foxes having a picnic in a magical forest, with tiny teacups, mushroom tables, and glowing fireflies. Style: Whimsical children's book illustration."

out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]
out.save("4bit.png")
```

Let's examine the result and check VRAM usage:

![Image generated](/assets/images/posts/2025-06-09-image-generation-example3.png)

**Peak memory usage: 19.05 GB**
**Generation time: 28 seconds**

As you can see, it generated an image quite similar to the previous one, but with some slight differences. In terms of quality, it's very similar to the previous results.

## Combining CPU Offload and NF4 Quantization

What if we combine CPU offload and NF4 quantization?

To do so we need to enable it on the pipeline (exactly like we did before):

```python
pipe.enable_model_cpu_offload()
```

- **Peak memory usage: ~8.5 GB**
- **Generation time: 39 seconds**
- **Result: Identical image quality**

This combination provides excellent memory efficiency while maintaining reasonable performance.

## Performance Summary

Here's a comparison of all the techniques we've explored:

| Technique | VRAM Usage | Generation Time | Image Quality | Use Case |
|-----------|------------|-----------------|---------------|----------|
| Full Precision | 39.27 GB | 26 seconds | Reference | High-end GPUs only |
| Sequential CPU Offload | ~2 GB | 5+ minutes | Identical | Very low VRAM |
| Model CPU Offload | ~24 GB | 56 seconds | Identical | Balanced approach |
| Manual Loading | 29.68 GB | 25 seconds | Slightly different | Scalable architectures |
| NF4 Quantization | 19.05 GB | 28 seconds | Very similar | Good balance |
| NF4 + CPU Offload | ~8.5 GB | 39 seconds | Very similar | Mid-range GPUs |

## Key Takeaways

- **Full precision with sequential pipeline**: Choose this if you have abundant VRAM and want the fastest generation
- **CPU offloading**: Use when running on very low VRAM systems, accepting significantly higher latency
- **Model offloading**: Offers much faster inference than sequential offloading with substantially lower VRAM usage
- **Manual module loading**: Excellent for scalable architectures where you can decouple modules across different compute resources
- **NF4 quantization**: Provides excellent quality and speed with reasonable VRAM requirements-check if quality meets your specific use case
- **Combined approaches**: NF4 + CPU offload offers the best memory efficiency for mid-range GPUs

The choice between these techniques depends on your hardware constraints, quality requirements, and latency tolerance. For most users with modern mid-range GPUs (12-16GB VRAM), NF4 quantization alone or combined with model CPU offload provides the best balance.

## Additional Resources

- [Diffusers Memory Optimization Guide](https://huggingface.co/docs/diffusers/en/optimization/memory)
- [Diffusers Quantization Blog Post](https://huggingface.co/blog/diffusers-quantization)
- [Quantization Fundamentals Course](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
- [Quantization in Depth Course](https://www.deeplearning.ai/short-courses/quantization-in-depth/)
