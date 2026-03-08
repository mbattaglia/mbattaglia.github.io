---
layout: post
title:  "Transformers from First Principles: Part 2 - Embeddings"
author: Matías Battaglia
excerpt: |
  <div class="excerpt-container">
    <div class="excerpt-image">
      <img src="/assets/images/posts/transformers/embeddings.gif" alt="Embedding calculation animation">
    </div>
    <div class="excerpt-text">
        A deep dive into how transformers convert token IDs into numerical representations through embeddings. We cover token embeddings, positional embeddings, the embedding dimension tradeoff, and why models struggle at the edges of their context window. This is the second post in a series explaining transformers.
    </div>
  </div>
---
On [my previous post]({% post_url 2026-02-14-Transformers-tokenization %}), I explained what transformers are, and dived deep into the tokenization process. Let's continue with the series on the transformers architecture, this time I will talk about *Embeddings*.

Now we have a mapping of tokens to their integer IDs, and need to convert them into the numerical representations used in the internal operations within Transformers. These numerical representations are called *embeddings*.

Since token IDs are already numbers, you may be wondering why can't we just use them directly? The problem is that token IDs are arbitrary: token 464 isn't "closer" in meaning to token 465 than it is to token 10356. They're just lookup indices, with no relationships between them. We need a representation where numerical similarity reflects semantic similarity, and that's what embeddings provide.

## Embeddings

If I had to summarize what transformers do at an absurd level: they generate embeddings of their inputs, and iteratively update their values to better represent their context and meaning. Transformers then use this to predict what is more likely to be the next token in the sequence.

An embedding is a vector that represents a token in a high-dimensional space. If you read my previous [post on Retrieval Augmented Generation]({% post_url 2025-08-20-Building-a-RAG-from-scratch %}), you may remember the concept. The main point is that vectors for similar concepts and meanings will have numerically similar vectors. As an example, "cat" and "dog" will be represented with numerical vectors that are closer to each other than the vectors between "cat" and "refrigerator."

Quoting my previous post:
> The magic thing is that if we take these vectors and represent them in a multi-dimensional space, we will find that concepts that are semantically similar will be "close" within this space, and concepts that are semantically different will be far away.

![Embedding representation](/assets/images/posts/2025-08-25-vectors.gif)

If you come from classical ML, this will feel familiar: token IDs are essentially **nominal categorical variables**, where numbers are used as labels, with no meaningful order or distance between them. It's the same problem you encounter when encoding categories like colour names as integers ("Blue = 1", "Red = 2" doesn't mean Blue is "more" than Red). In traditional ML, you'd solve this with techniques like one-hot encoding. Transformers solve it with learned embeddings: a dense, continuous representation for each category (which is the same idea, just at a much larger scale).

In this section of the Transformer, we compute the embeddings that will be used as the starting representation for the neural network, by calculating the embedding for each position in the input sequence. It does so with the combination of two components: _what_ the token is (token embedding) and _where_ it appears (positional embedding).

### Token Embeddings
Each token in the vocabulary has a corresponding vector stored in an **embedding matrix**. This matrix has shape `(vocab_size, embedding_dim)`, where:

- `vocab_size` is the number of tokens in the vocabulary (e.g., 50,000)
- `embedding_dim` is the size of each embedding vector (e.g., 768, 1024, or 4096)

The token ID is used to look up the corresponding row in this matrix. For example, token ID 464 retrieves row 464 (a vector of `embedding_dim` numbers) from the embedding matrix.

```
Token ID: 464 → Embedding: [0.12, -0.34, 0.78, ..., 0.45]  # embedding_dim numbers
```

The embedding values in the embedding matrix are learned during training through backpropagation, just like any other weight in the network. The model discovers which numerical representations work best for predicting the next token.

#### What Does the Embedding Dimension Represent?
The embedding dimension determines how much information each token can carry. You can think of each dimension as a "feature" the model can use to encode meaning, though these features aren't normally interpretable.

To build some intuition: some dimensions might implicitly capture things like whether a token is a verb or a noun, whether it carries a positive or negative sentiment, or whether the word is formal or informal. The model figures out what "features" are useful during training and they are not chosen manually (though the idea of someone manually assigning a dimension to represent the concept of a Dachshund sounds hilarious).

In practice, individual dimensions aren't directly interpretable because embeddings are *distributed representations*: meaning isn't stored in any single dimension, but spread across many dimensions simultaneously. The concept of "animal" might be partially encoded across dozens of dimensions, each of which also participates in encoding other concepts. This is what makes them powerful (they can represent far more concepts than they have dimensions) but also what makes them opaque. If you are interested in the ongoing work to interpret these representations, I'd recommend reading [this paper](https://www.anthropic.com/research/mapping-mind-language-model) and [this paper](https://www.anthropic.com/research/tracing-thoughts-language-model) from Anthropic.

The embedding dimension is a key model design decision: larger dimensions allow the model to capture more nuanced relationships, but increase memory usage and computation. For reference, GPT-2 used 768 dimensions, while GPT-3 scaled to 12,288. This is partly what I meant in my previous post when I said larger vocabularies increase memory, as embedding dimension matters just as much. The embedding matrix for GPT-2's configuration (50,257 tokens × 768 dimensions) contains ~38.6 million parameters. Scale the dimensions to GPT-3's 12,288 and you're looking at ~617 million parameters, just for the embedding matrix alone.

### Positional Embeddings
A limitation of the token embeddings is that they do not encode position in any way, as the same token will be assigned the same embedding value regardless of where it is located in a sequence. This is a problem, as position is important while modelling language: 'the cat sat on the mat' and 'the mat sat on the cat' would look identical to the model, though those convey different ideas. Long story short, there needs to be a way to encode positioning into the embeddings.

This is specifically a problem for transformers because, unlike older architectures like RNNs or LSTMs that process tokens one after the other (and therefore have an inherent notion of order), transformers process all tokens in the sequence simultaneously. This parallelism is what makes them fast, but it means they have no built-in sense of position.

This is achieved through **positional embeddings**, in which each position in the sequence gets its own vector. These are stored in a separate **positional embedding matrix** of shape `(max_sequence_length, embedding_dim)`, where `max_sequence_length` is the longest sequence the model can handle (e.g., 1024 for GPT-2). This matrix is also learned during training, just like the token embeddings.

The lookup works the same way: position 0 retrieves row 0, position 1 retrieves row 1, and so on:

```
Position: 0 → Positional Embedding: [0.02, 0.15, -0.03, ..., 0.11]  # embedding_dim numbers
Position: 1 → Positional Embedding: [0.08, -0.22, 0.14, ..., 0.07]  # embedding_dim numbers
```

The key difference from token embeddings is that the index is the position within the sequence, not a token ID. The token "cat" always maps to the same row in the token embedding matrix, but it gets a different positional embedding depending on whether it appears at position 3 or position 50.

#### The edge of the context window

Since positional embeddings are learned, they are only as good as the training data they were exposed to. During training, models are exposed to shorter sequences far more frequently than longer ones, meaning that the model has significantly more "experience" with earlier positions than with later ones, and the quality of the positional representations degrades as you approach the edges of the context window.

This has a hard limit too: learned positional embeddings have a fixed `max_sequence_length`, meaning the model has literally never seen positions beyond that boundary. If you try to extrapolate, the model breaks. This is one of the reasons why techniques like [RoPE](https://arxiv.org/abs/2104.09864) (Rotary Position Embeddings) were developed. This type of embedding encodes position mathematically rather than through learned vectors, allowing models to generalise to longer sequences than those seen during training. RoPE embeddings encode position as a *rotation* in the embedding space: each position rotates the embedding vector by a different angle, so the model can infer relative distances between tokens from the angle between their rotated vectors. Since rotations are continuous and periodic, the model can extrapolate to positions it hasn't seen, which learned positional embeddings simply can't do.

Even with these techniques, the core insight I made at the beginning of this series remains: models perform better with shorter, focused context.

### Combining Token + Position
The final input embedding for each position is the sum of the token embedding and the positional embedding:

```
input_embedding[i] = token_embedding[token_id[i]] + positional_embedding[i]
```

This element-wise addition produces a single vector per position that encodes both _what_ the token is and _where_ it appears.

You might wonder why simple addition works here rather than, say, concatenation. Addition works because the embedding space is high-dimensional, giving the model plenty of room to learn complementary subspaces for identity and position information. The two signals don't destructively interfere because they effectively occupy different regions of the space.

<video autoplay loop muted playsinline width="100%">
  <source src="/assets/images/posts/2026-02-14-transformers/embedding.mp4" type="video/mp4">
</video>

### Embeddings Input and Output
To summarize what happens in this section: the embedding block takes a list of token IDs as input and produces a matrix of shape `(sequence_length, embedding_dim)` as output. Each row is the combined token + positional embedding for that position.

Taking our running example from the previous post:
```
Input:  [464, 3797, 3332, 319, 262, 15488]    # "The cat sat on the mat"
↓
Output: [[0.14, -0.19, 0.81, ..., 0.52],      # position 0: "The"
        [0.33,  0.45, -0.12, ..., 0.18],      # position 1: "cat"
        [0.27, -0.08,  0.64, ..., 0.41],      # position 2: "sat"
        [0.09,  0.55, -0.33, ..., 0.29],      # position 3: "on"
        [0.16, -0.07,  0.78, ..., 0.58],      # position 4: "the"
        [0.71,  0.22,  0.03, ..., 0.67]]      # position 5: "mat"
# shape: (6, embedding_dim)
```

Note that token IDs 464 ("The") at position 0 and 262 ("the") at position 4 have different token IDs (capitalisation matters in tokenization), but even if they shared the same token ID, their final embeddings would differ because they receive different positional embeddings.

This matrix is the input for the transformer layers.

## What's Next
We now have the inputs encoded as embeddings, it's time to pass those along to the **transformer layers**. In the next post, we will explore what the **attention mechanism** is, and how it updates the embeddings to better represent the semantics of the input.