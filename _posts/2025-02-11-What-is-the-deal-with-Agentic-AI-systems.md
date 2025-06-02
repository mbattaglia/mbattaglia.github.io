---
layout: post
title:  "What is the deal with Agentic AI systems?"
author: Matías Battaglia
excerpt: |
  <div class="excerpt-container">
    <div class="excerpt-image">
      <img src="/assets/images/posts/2025-02-11-agentic-ai.png" alt="Agentic AIg">
    </div>
    <div class="excerpt-text">
        Are agentic AI systems always the answer? In this post, I explore why simplicity often beats complexity, when agent-based architectures make sense, and why you should reach for agents only when truly needed.
    </div>
  </div>

---

![](/assets/images/posts/2025-02-11-agentic-ai.png)

[(originally posted on LinkedIn)](https://www.linkedin.com/pulse/what-deal-agentic-ai-systems-matias-battaglia-romano-ermnf/)

There's a lot of hype circulating about using agents to solve pretty much every use case under the sun. There's extensive discussion and content about multi-agent systems, orchestration and coordination layers, and much more.

While this approach will certainly be the right solution for some advanced use cases (and I'll share my opinion on when those solutions make sense later), this current trend makes it look like the only valid solution is to build these AI Rube Goldberg machines (chef's kiss meme to Claude 3.5 Sonnet for the lovely analogy).

## Unnecesary complexity
Complexity creates barriers for organizations and individuals taking their first steps in the AI world. It's my duty as a technologist to stand against unnecessary complexity and to preach for simpler, easier-to-understand systems and patterns.

In [their publication](https://www.anthropic.com/research/building-effective-agents) on agentic AI, Anthropic analyzed successful real-world implementations and found:

> Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

I think this quote pretty much nails it - it's hard to argue with this; it's even harder to argue with a company such as Anthropic with its experience helping organizations build "real world" stuff with their products.

I **LOVE** the idea of having simpler, composable systems that you can use as building blocks to build more complex solutions (maybe this should not come as a surprise since I work for AWS).

## Agentic patterns
I see 3 common approaches to using LLMs to solve use cases, in order of increasing complexity:

- **Directly invoking LLMs**: simply calling an LLM and getting a response. Nothing else.
- **Workflows**: systems with predefined paths and flow control. "If this, then that". LLMs are "just" elements within those paths, not controlling the execution flow.
- **Agents**: systems where the LLM dynamically controls the flow of the execution, potentially invoking other agents or even choosing when to stop working after fulfilling or completing a task.

I will go a step further and say that many use cases can be solved with the first approach, then fewer require workflows, and even fewer truly need agents. These are opposite ends of a spectrum, with implementations that fall somewhere in between.

> But Matías, which approach should I use?
I hear you asking aloud somehow, even though this is a LinkedIn article and it's impossible for me to actually hear you (I guess I should say that I hear you metaphorically).

Well, the answer is that **it depends** (took me long enough to say it).

As I mentioned before, you should aim to use the simplest solution possible, and only increase complexity if strictly needed.


Each approach has its own set of advantages:

- **Directly calling LLMs**: simpler, quicker approach. Perfect for straightforward tasks where a single response is enough. While it may not handle all use cases or situations requiring multiple coordinated steps, it's often the right choice for many common scenarios.
- **Workflows**: great for well-defined tasks with clear steps. They offer predictability and consistency, making them perfect when reliability and repetibility are key. They're also easier to test and debug since you know exactly what path the execution will follow.
- **Agents**: best used for complex, open-ended tasks where flexibility and dynamic decision-making are crucial. They can handle unpredictable scenarios and adapt their approach based on intermediate results, though they come with higher costs and complexity.


## Multi-agent and other more complex approaches
I started this talking about "unnecesary complexity", yet I believe some use cases may require multiple agents working alongside (or may even require more complex approaches).

There are use cases that are asynchronous by nature, requiring work to be sent to a worker and then fetching the results later.

There are use cases that cross multiple business domains; requiring coordination between agents specialized in different tasks.

I'll cover this more complex approaches in a later post.

## Takeaways
- Start simple by default; add complexity only when it demonstrably improves outcomes.
- If a single LLM call solves the problem, that’s enough.
- Agents should be the exception, not the default. 
- When multi-agent systems are necessary, ensure they provide clear value beyond a simpler alternative.

Your users (and future maintainers) will thank you for it. 🚀
