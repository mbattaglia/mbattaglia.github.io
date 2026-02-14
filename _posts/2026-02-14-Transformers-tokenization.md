---
layout: post
title:  "Transformers from First Principles: Part 1 - Tokenization"
author: Matías Battaglia
excerpt: |
  <div class="excerpt-container">
    <div class="excerpt-image">
      <img src="/assets/images/posts/transformers/transformers-faster.gif" alt="An animation explaining the workflow of a transformer.">
    </div>
    <div class="excerpt-text">
        A deep dive into how transformers process text, starting with the first step: tokenization. We cover vocabularies, the tradeoffs of vocabulary size, special tokens, and edge cases like anomalous tokens. This is the first post in a series explaining transformers.
    </div>
  </div>
---
# Transformers from First Principles: Part 1 - Tokenization

When using tools like Claude Code, have you noticed how important it is to keep the context window clean of irrelevant data and focused on a specific task? Empirically, you quickly learn that proper context management not only improves performance, it also lowers latency and saves on tokens.

But _why_? Have you ever wondered _why_ that is?

In this post I will kick off a series of posts where I will explain how LLMs generate text and why this limitation is inherent to them. I am writing this because I am a big fan of learning from _first principles_, and I have the hope that it will help build your intuition.

# What are transformers?
To understand how LLMs generate text, you need to understand the transformer architecture, as it is the underlying architecture used in most LLMs. It's a type of neural network designed to work with sequences, predicting what comes next by considering what came before.

As a heads up, in this post series I will focus on the specific transformer variant used in "GPT-2 style" models, which specialize in next-token prediction (I will explain what a token is in the next section). While modern LLMs have evolved beyond this original architecture, its core building blocks still power them today.

Transformers are **autoregressive**, meaning they generate output one token at a time, feeding each prediction back into the input to generate the next one. This loop continues until the model produces a stop token or reaches a configured maximum length.

![Transformers animation](/assets/images/posts/transformers/transformers.mp4)

Let's explore the main conceptual blocks in a transformer.

# The Anatomy of a transformer
Let's start by understanding what the actual input is to transformers. While you do send text (or images, audio, etc) to LLM's APIs, in reality the actual input to LLMs is converted into an array of integers, which LLMs then convert into floating-point vectors and process them through matrix operations, finally outputting a probability distribution over their vocabulary. So there is no text at any point.

At a conceptual level, a transformer is composed of the following blocks:

1) Tokenization
2) Embedding
3) Transformer layers
4) Unembedding
5) Sampling

Or, if you prefer an animation:

![Transformers pipeline animation](/assets/images/posts/transformers/transformers-pipeline.mp4)

Let's walk through each one. In this post, I will focus on tokenization, and the remaining blocks will be covered in subsequent posts.

## Tokenization
Since transformers don't work with text, there needs to be a way to convert the input text into a numerical representation. This is done by splitting the text into smaller chunks called _tokens_, and then mapping each token to its corresponding numerical _token ID_ following a _vocabulary_.

Tokens can be common words, word fragments, punctuation, emojis, or special tokens like end-of-sequence, or tool-use markers. For example, "transformers" might become `["trans", "form", "ers"]`. Usually, different model providers or labs have different tokenizers, and different generations of models may have different tokenizers as well. 

![Tokenization animation](/assets/images/posts/transformers/tokenization.mp4)

The result of tokenization is an array of integers, where each integer is an index into the vocabulary:

```
"The cat sat on the mat" -> [464, 3797, 3332, 319, 262, 15488]
```

When processing multiple sequences at once (batching), this becomes a 2D tensor of shape `(batch_size, sequence_length)`.

### Vocabulary
A vocabulary is the complete set of tokens a model can recognize and produce. A model _cannot_ process a token outside its vocabulary (though it can disassemble an unknown word into known tokens for its input and do the opposite for its output).

While tokenization itself is a simple lookup process, building the vocabulary is not.

#### The vocabulary size tradeoff
The size of the vocabulary is one of the most important parameters of a model, and it involves a fundamental tradeoff:

- **Larger vocabulary**: With a larger vocabulary, common words and phrases can become single tokens, effectively meaning that you will use fewer tokens to encode an arbitrary input  (and therefore increasing the efficiency of the context window). This is at the expense of the embedding and unembedding matrices growing larger,  increasing memory usage and computation (we will see what this means in a later post). A larger vocabulary also allows you to natively encode characters from other languages.

- **Smaller vocabulary**: A smaller vocabulary means that you will have to use smaller tokens (such as individual characters, small pieces of words, etc) to encode words, meaning that you will need to use more tokens to encode an arbitrary input. This means that your context window is used less efficiently, but your model will have a lower number of total parameters, decreasing memory usage.

For example, with a large enough vocabulary you may have a single token to represent the word "transformers". But with a small vocabulary, it might become `["t", "ran", "s", "form", "ers"]`, taking five tokens instead of one. Multiply this across an entire document and your sequence length explodes.

Vocabulary size also affects hardware efficiency. Matrix operations on GPUs are optimized for specific matrix dimensions, and a non-optimized vocabulary size can significantly impact performance. Andrej Karpathy discusses this in detail in [this post](https://x.com/karpathy/status/1621578354024677377).

#### Special Tokens
Beyond regular text tokens, vocabularies include special tokens that control model behavior:

- `<|endoftext|>`: Marks the end of a sequence
- `<|startoftext|>`: Marks the beginning of a sequence
- `<|pad|>`: Used to pad shorter sequences in a batch to equal length
- Tool-use tokens: Some models include special tokens for function calling or structured outputs

These tokens are treated just like any other token internally, but they carry special meaning during training and inference.

#### Weird edge cases of Tokenization
Now to the fun stuff: tokenization opens the door to interesting edge cases. You may remember the issue LLMs had when counting the number of letter "r"s in the word strawberry; this was a side effect of tokenization.

Another example is: what happens with tokens that exist in the vocabulary but rarely (or never) appeared in the training data? To make an analogy, imagine that you know how to say a word in a foreign language but you have absolutely 0 idea what it means - there is no way you can use that word effectively!

The same situation happens with LLMs, and is known as "anomalous tokens". Those cause unpredictable behavior in models, producing nonsensical outputs or making the model behave erratically. This has been the source of some well-documented strange behaviors in [early ChatGPT versions](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation), or [more recently with DeepSeek-V3](https://outsidetext.substack.com/p/anomalous-tokens-in-deepseek-v3-and).

### Tokenization Input and Output
Summarizing, the function of tokenization is to take text as input and generate a list of integers as its output.

#### A note about shapes and batching
**Batching** is processing multiple sequences simultaneously instead of one at a time. Rather than feeding the model a single sequence and waiting for the output, you group several sequences together and process them in parallel. This is significantly more efficient because GPUs are optimized for parallel operations, meaning that processing 32 sequences at once isn't 32x slower than processing one.

When batching, tensors gain an extra dimension. A single sequence has shape `(sequence_length,)`, but a batch has shape `(batch_size, sequence_length)`. For example, a batch of 32 sequences, each with 100 tokens, would be a tensor of shape `(32, 100)`.

For this post series, I will ignore batching to make the numbers easier to understand. If you want to add `batch_size`to any of the shapes mentioned, you can add an extra dimension as described in the previous paragraph.

One curiosity about batching is that it is one of the main reasons why any optimized LLM inference server produces non-deterministic results. You can read more about it on [this amazing blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) from Thinking Machines.

## What's Next
We have seen how text becomes integers through tokenization. But a set of discrete integers can't be used in neural network computations. In the next post, we will explore how these token IDs become numerical representations called **embeddings**.