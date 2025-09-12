---
layout: post
title:  "Using Structured Output with LLMs"
author: Matías Battaglia
excerpt: |
  <div class="excerpt-container">
    <div class="excerpt-image">
      <img src="/assets/images/posts/2025-09-11-structured-output.gif" alt="Structured output from LLMs">
    </div>
    <div class="excerpt-text">
        This post covers three approaches for getting structured output out of LLMs: use native structured output APIs when available, fall back to function calling for broader provider support, or combine careful prompting with validation as a universal solution.
    </div>
  </div>
---

Structured output from LLMs is critical for production applications, yet it's one of those topics that seems simple in proof-of-concept mode but suddenly becomes a problem at scale.

Let's explore different approaches to reliably getting structured output from LLMs.

![Structured output](/assets/images/posts/2025-09-11-structured-output.gif)

## What do I mean by "using structured output with LLMs"?

By this I mean defining a JSON model or schema for how outputs from LLMs should look like, and then coercing the LLM to follow it. It also covers the mechanisms you can use to validate whether the LLM did it, and even casting them if needed.

You can also view it as applying strong typing to LLM outputs.

## Why would you care about this?

The short answer: **downstream integrations**. If you are planning to integrate the LLM's output with any other system, most likely you will want to have control over the structure and the fields contained in the output.

Have you ever prompted an LLM to answer only with a JSON and in response you got those dreaded markdown code ticks (` ```json `)? Without structured outputs, you're essentially playing a game of chance. You might get perfect JSON 99% of the time, but that 1% failure rate will break your system.

## How does it work?

There are multiple ways to make it work, depending on the libraries and the LLMs you are using.

You first need to define a model for your data. I prefer to use [Pydantic](https://docs.pydantic.dev/latest/) for its extensibility, its strong validation and type casting/conversion and how well integrated it is across the LLM landscape, but you could also define a JSON schema directly.

A model could look like this:

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str = Field(..., pattern=r'^\d{5}(-\d{4})?$')
    country: str = "ES"
```

**Tip: Start simple**: Start with flat structures before moving to nested ones.

Many LLM providers include a parameter in their API to pass along a model ([OpenAI](https://platform.openai.com/docs/guides/structured-outputs), [Gemini](https://ai.google.dev/gemini-api/docs/structured-output), [Ollama](https://ollama.com/blog/structured-outputs), etc). The provider will abstract away the format validation and the retry logic on failure.

Here's how it looks with OpenAI:

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "Extract the address."},
        {
            "role": "user",
            "content": "123 Main St, Madrid, Spain 28001",
        },
    ],
    text_format=Address
)

address = response.output_parsed
print(f"Address: {address}")
```

But not all providers offer this feature natively (as of September 2025, this list includes services like AWS Bedrock, Anthropic, etc).

## The Manual Approach nº1: Prompting for Structure

To get structured output from models that don't support it natively, you need to prompt the LLM to answer using your schema. You can try something like this:

```python
import json

schema = Address.model_json_schema() # get JSON schema from the previously defined model

system_prompt = f"""You are a helpful assistant that always responds with valid JSON.
    
Your response MUST follow to this exact JSON schema:
{json.dumps(schema, indent=2)}
    
Do not include markdown code blocks, or any explanatory text before or after the JSON.
Ensure all required fields are present"""
```

It's a good idea to provide examples in your prompt of the specific format you are expecting. You can also use [Pydantic validators](https://docs.pydantic.dev/latest/concepts/validators/) to, as the name implies, validate any custom field you may need to use.

For a deeper understanding on this pattern, I highly recommend the [Pydantic for LLM Workflows](https://www.deeplearning.ai/short-courses/pydantic-for-llm-workflows/) short course in [DeepLearning.ai](https://www.deeplearning.ai/).

## Function Calling: Another tool in the toolbox

Keen readers may be wondering whether you could use native function calling to achieve this - and you would be totally right! This is actually one of the most reliable ways to get structured outputs from models that support function calling. Anthropic even [calls out](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview#json-mode) this technique.

The trick is to define a function that takes your desired output as parameters:

```python
from openai import OpenAI, pydantic_function_tool

client = OpenAI()

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "Extract the address."},
        {
            "role": "user",
            "content": "123 Main St, Madrid, Spain 28001",
        },
    ],
    tools=[pydantic_function_tool(Address)]
)

```

You dont even need to define an actual function; it's just enough to pass along its schema/model. At the end of the day, function calling heavily conditions the model to answer with a JSON (LLMs are usually trained to be very precise with function arguments).

This approach is generally more reliable than free-form JSON generation and saves you from prompting asking for a prompt.

## Advanced Techniques

### Prefilling

For models like Claude, you can use a technique called [prefilling](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response#example-structured-data-extraction-with-prefilling) to increase reliability. Prefilling is like "putting the words in the LLM's mouth", meaning that you "pre-write" the LLM response in the assistant's turn within the messagest list. Let's see an example:

```python
# Prefill the assistant's response to start with valid JSON
messages = [
{"role": "user", "content": f"Extract the address from: 123 Main St, Madrid, Spain 28001"},
    {"role": "assistant", "content": "{"}  # we are "pre-writing" the start of the JSON
]
# The model will continue from where you left off
```

### Constrained Generation

Some libraries like [Outlines](https://github.com/outlines-dev/outlines) or [Guidance](https://github.com/guidance-ai/guidance) take this even further by constraining token generation to only produce valid JSON:

```python
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.json(model, Address)
result = generator("Extract address: 123 Main St, Madrid, Spain")
# Guaranteed to return valid Address object or fail
```

## Implementation Strategy

Here's how to approach structured output in practice:

- **First choice**: Use native structured output (OpenAI, Gemini) when available. This has the highest reliability and requires the least effort.
- **Second choice**: Use function calling for providers that don't support structured output natively. It's nearly as reliable as native support, but requires more effort to implement.
- **Fallback**: Combine prompting with custom validation for pretty much any provider and model combination.
- **Always implement**: Monitor success rates in production and have fallback plans for critical applications.

**For advanced use cases**: Consider constrained generation libraries when you need guaranteed validity.