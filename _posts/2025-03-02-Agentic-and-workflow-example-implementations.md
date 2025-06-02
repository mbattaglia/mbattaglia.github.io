---
layout: post
title:  "Agentic and workflows example implementations"
author: Matías Battaglia
excerpt: |
  <div class="excerpt-container">
    <div class="excerpt-image">
      <img src="/assets/images/posts/2025-03-02-workflow-routing.png" alt="Workflow Example">
    </div>
    <div class="excerpt-text">
      Practical examples of agentic and workflow-based AI patterns, with code and design decisions inspired by Anthropic’s research. Learn how to implement direct LLM calls, prompt chaining, and routing workflows—while keeping your systems simple and maintainable.
    </div>
  </div>
---

I've implemented the patterns described in Anthropic's "[Building effective agents](https://www.anthropic.com/research/building-effective-agents)" blogpost and want to share the key design decisions and lessons learned along the way. This post breaks down the practical considerations for implementing effective AI systems without unnecessary complexity.

Be mindful this are just example implementations, not meant for production usage.

## Main Features

- **Implementation of multiple patterns**: Direct LLM invocation, prompt chaining workflows, and routing workflows
- **Schema enforcement**: Using Pydantic and LangChain to ensure consistent, parseable LLM outputs
- **Minimal dependencies**: Only essential libraries are used to maintain simplicity
- **Focus on composability**: Building blocks that can be assembled into more complex systems when needed

## Some Context

[Some time ago]({% post_url 2025-02-11-What-is-the-deal-with-Agentic-AI-systems %}), I wrote about the excessive hype surrounding AI agents and advocated for a layered approach when designing AI-powered solutions. My core belief is that organizations should maximize simplicity and composability, only adding complexity when absolutely necessary.

In that post, I outlined three general implementation patterns, ordered by increasing complexity:

| Pattern | Description | Best For | Complexity |
|---------|-------------|----------|------------|
| **Direct LLM Invocation** | Simply calling an LLM and getting a response | Straightforward tasks with clear inputs | Low |
| **Workflows** | Systems with predefined paths and flow control | Well-defined processes with conditional logic | Medium |
| **Agents** | Systems where the LLM controls execution flow | Open-ended, complex tasks requiring adaptation | High |

## The Repo

I've created [a GitHub repository](https://github.com/mbattaglia/building-effective-agents) that implements the first two approaches. I'm currently working to extend it with more examples of workflows and agent use cases.

So far, I've implemented:

### 1. Directly Calling LLMs
![Workflow Direct Invocation](/assets/images/posts/2025-03-02-workflow-direct-invocation.png)

This pattern directly calls an LLM.

### 2. Workflow: Prompt Chaining
![Workflow Prompt Chaining](/assets/images/posts/2025-03-02-workflow-prompt-chaining.png)

This pattern passes the output of one LLM call as input to another, creating a pipeline of transformations.

### 3. Workflow: Routing
![Workflow Routing](/assets/images/posts/2025-03-02-workflow-routing.png)

This approach uses an LLM to determine which path in a workflow to take, then routes the request accordingly. It's particularly useful when you need conditional logic based on content analysis.

## Design Decisions and Tradeoffs

### Forcing Schema on LLM Outputs

I'm using LangChain to "force" LLMs to generate output following specific schemas. I think this is a great approach, as it  offers multiple benefits:

- **Easier integration**: Structured outputs are simple to parse and integrate with downstream systems
- **Validation**: Pydantic ensures responses can be parsed into their respective schemas
- **Consistency**: Reduces unexpected output formats that could break your application

```python
# Example schema definition
class Section(BaseModel):
    title: str = Field(
        description="Title of the section",
        min_length = 5,
        max_length = 100
    )
    
# Force the LLM to use this schema
structured_llm = llm.with_structured_output(Section)
```

This is particularly effective with newer LLMs that have been trained to leverage tools or have a native "JSON mode".

### Use of 3rd Party Libraries

I've intentionally kept dependencies minimal, using only:

- **LangChain**: For abstracting LLM integrations, and forcing schema validation
- **LangGraph**: For building and visualizing workflows
- **Pydantic**: For schema validation and type safety

### Concurrency: When Less is More

While some functions implement concurrency, I deliberately avoided it in certain places like LLM invocation. The reason is practical:

> I am using an LLM running locally and my PC doesn't have the horsepower to run multiple inference pipelines in parallel.

While this is anecdotical, I think it also highlights something that I see when working with real-world use cases: theoretical optimizations sometimes don't align with practical constraints.

## What's Next

I'm currently working on extending the repository with:

1. More complex workflow examples (parallelization, "Orchestrator-Worker", "Evaluator-optimizer", etc.)
2. Agent implementations using LangGraph
3. Performance comparisons between different approaches

## Final Thoughts 💡

Building effective AI systems doesn't require complex agent architectures for every use case. By following a layered approach and prioritizing simplicity, you can create maintainable, understandable systems that deliver real value.

The most important lesson I've learned is that the right level of abstraction depends entirely on your specific requirements. Don't add complexity for its own sake - let the problem guide your technical choices.