# Short readings for software practice, LLM fine-tuning, and DKT

- **Prepared:** 2026-10-11
- **Audience:** Students who already program and use GitHub, but may be new to professional software work or machine learning.
- **Scope:** A selective starting path for team software development, transformer fine-tuning, and the neural-network concepts needed for Deep Knowledge Tracing (DKT).

## Software development as team work

1. **What changes when code becomes a team product:** Read the opening discussion in Google's *Software Engineering at Google*, [Chapter 1: “What Is Software Engineering?”](https://abseil.io/resources/swe-book/html/ch01.html). In [Chapter 2: “How to Work Well on Teams”](https://abseil.io/resources/swe-book/html/ch02.html), read “The Genius Myth” and “It’s All About the Team.” These are useful industry perspectives from Google, not universal rules for every organization.
2. **How contributors make work reviewable:** Google's short [Small CLs](https://google.github.io/eng-practices/review/developer/small-cls.html), [Writing good CL descriptions](https://google.github.io/eng-practices/review/developer/cl-descriptions.html), and [Standard of Code Review](https://google.github.io/eng-practices/review/reviewer/standard.html) guides show how to divide work into reviewable changes, explain intent, and use review to improve shared code. Google calls a change a “CL”; read that as a pull request or proposed change in this project.

## ML concepts for DKT and LLM fine-tuning

This path skips broad ML surveys and framework tutorials. Read only the sections needed for the work:

1. **Neural networks and learning:** 3Blue1Brown's [But what is a Neural Network?](https://www.3blue1brown.com/lessons/neural-networks/), [Gradient descent](https://www.3blue1brown.com/lessons/gradient-descent/), and [What is backpropagation really doing?](https://www.3blue1brown.com/lessons/backpropagation/) explain layers, weights, loss, and how training updates weights. The backpropagation calculus follow-up is optional.
2. **Transformer models:** 3Blue1Brown's [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt/) and [Attention in transformers](https://www.3blue1brown.com/lessons/attention/) give the architecture and attention intuition needed to understand which model parameters fine-tuning changes.
3. **Sequence models and DKT:** Read Chris Olah's [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) for recurrent hidden states and gates. Then read the abstract, Figure 2, and Section 3.1 (“Model”) of the original [Deep Knowledge Tracing paper](https://papers.nips.cc/paper_files/paper/2015/file/bac9162b47c56fc8a4d2a519803d51b3-Paper.pdf). The paper shows how a recurrent model uses a student's interaction history to predict future responses.
4. **SFT and LoRA:** In Hugging Face's Smol Course, read [Supervised Fine-Tuning](https://huggingface.co/learn/smol-course/en/unit1/3) and [LoRA and PEFT](https://huggingface.co/learn/smol-course/en/unit1/3a). Focus on SFT's data shape and quality, held-out evaluation and overfitting, and why LoRA updates small adapter weights while the base model is frozen; skip setup and code exercises until a training task is selected.
5. **DPO:** Read Hugging Face's [Direct Preference Optimization](https://huggingface.co/learn/smol-course/en/unit2/2) after SFT. It explains training from chosen/rejected response pairs. DPO is preference optimization and has a different training objective from policy-gradient RL.
6. **RL for LLMs:** Read Hugging Face's [Introduction to Reinforcement Learning and its Role in LLMs](https://huggingface.co/learn/llm-course/en/chapter12/2) for the policy, reward, and RLHF/GRPO picture. If the project will use GRPO, add “What is RL?”, “How GRPO Trains a Model,” and “Reward Functions / Verifiers” in Unsloth's [Reinforcement Learning guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide); treat those sections as a practical method guide, not a general RL course.

Hugging Face's Smol Course assumes basic ML, a deep-learning framework, and transformer familiarity. Assign the linked concept sections first; its code exercises can wait until students have project-specific framework practice. Unsloth is most useful later as a hands-on training path; its implementation steps and current GPU requirements depend on the chosen model and hardware.
