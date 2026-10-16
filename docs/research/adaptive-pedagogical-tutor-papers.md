# Research notes: adaptive pedagogical tutors

- **Prepared:** 2026-10-08
- **Scope:** Decision-oriented, primary-source review of 40 selected papers for building an adaptive tutor for a student club. This is a curated, decision-relevant set rather than an exhaustive bibliography; math tutoring dominates the available benchmarks and RL studies. Repository had no existing research-note convention, so this note is saved under `docs/research/`.

## Quick read

These 40 papers span five decision areas: tutoring foundations and learning science; LLM tutoring behavior and alignment; learner-state modeling and adaptation; benchmarks and safety evaluation; and classroom or field evidence. The lowest-cost starting point for a student club is to specify tutor behavior, elicit learner explanations, capture only useful learner context with consent, and evaluate it against club-authored cases before considering fine-tuning or reinforcement learning.

Several studies are math-specific. Their dialogue structures and evaluation principles transfer to another club subject, but their math benchmarks, student simulators, and answer-checking rewards do not transfer unchanged.

The classic ITS evidence suggests design ideas worth carrying forward—step-level feedback, formative assessment, corrective practice, and learner self-explanation—but it does not show that an LLM tutor will reproduce the same outcomes. The newer intervention studies also underline that implementation and actual student engagement matter alongside model behavior.

## At-a-glance: decisions this literature can inform

| If you need to decide… | Start with… | Practical takeaway |
|---|---|---|
| How to avoid answer-dumping | LearnLM, MathDial, Intent Matters, Bastani et al. | Specify the next teaching move, and test whether students can solve a fresh problem without help. |
| What to remember about a learner | Knowledge Tracing, Adaptive Scaffolding, LongTutor | Track evidence by skill; separate history retrieval, knowledge diagnosis, and the next teaching action. |
| When and what hint to give | Alam et al., Chung et al., Atkinson et al., Oreopoulos et al. | Adapt to observed attempts, then use hints, worked examples, retries, and mastery checks as a loop. |
| Whether the tutor helps | Kestin et al., Nigeria RCT, Worden et al., Kraft et al. | Measure unaided learning and implementation; short-term answer correction is not the same as durable transfer. |
| Whether a model or simulator is trustworthy | MathTutorBench, TutorBench, simulated-student benchmark, SHAPE, answer-leakage benchmark | Evaluate separate skills, actual context sensitivity, safety, and adversarial pressure; do not use a simulator score as a learning outcome. |

## Paper summaries

### 1. PedagogicalRL — *From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning* (arXiv:2505.15607)

> **What you need to know (ELI5):** Multi-turn reinforcement learning can teach a tutor to guide a learner through a conversation while balancing progress with answer withholding. It is a later-stage recipe to consider only when the club can validate its simulated students and learning signal.

- **Problem:** Models trained to answer questions tend to reveal solutions; single-turn training misses how a tutor should adapt over a full dialogue.
- **Technology and solution:** Online, on-policy multi-turn RL in simulated student–tutor conversations, using Qwen2.5-7B as the tutor and an LLM student. GRPO updates the tutor from complete dialogue rollouts. A numerical post-dialog student solve rate is combined with pedagogical acceptance checks (including answer leakage and helpfulness), with a tunable penalty to trade off solving success against pedagogy. The post-dialog attempts are conditioned on the dialogue, so this metric alone does not establish tutor-free transfer.
- **What is distinctive:** It learns from the tutor’s own evolving dialogue episodes without human-annotated tutor responses and makes the pedagogy-versus-answer-success trade-off explicit. Optional `<think>` tags expose planning, but this paper does **not** reward the content of the thinking trace; that is the extension in paper 2.
- **Club use:** A later-stage recipe if the club has enough representative practice problems and can validate a simulated learner. Adapt the post-dialog success objective, but use a fresh problem without the tutor dialogue if you want to measure independent transfer. The paper’s answer-leakage measure is a useful companion metric.
- **Primary sources:** [arXiv paper and current v2](https://arxiv.org/abs/2505.15607); [official ETH-LRE code repository](https://github.com/eth-lre/PedagogicalRL).

### 2. PedagogicalRL-Thinking — *Rewarding How Models Think Pedagogically: Integrating Pedagogical Reasoning and Thinking Rewards for LLMs in Education* (arXiv:2601.14560)

> **What you need to know (ELI5):** This work rewards a model for planning its teaching approach, including using Pólya’s problem-solving steps, as well as for the tutoring dialogue it produces. The planning framework can inspire tutor prompts, but the study’s hidden reasoning reward is not itself evidence of student learning.

- **Problem:** Rewarding only the words shown to a student does not directly train a reasoning model to plan instruction around that student’s understanding.
- **Technology and solution:** Multi-turn RL with GRPO in veRL. It prompts the model to use Pólya’s four steps (understand the problem, devise a plan, carry it out, look back), and adds a GPT-4o-mini judge score for pedagogical quality of the model’s `<think>` trace. The total reward combines student solve-rate improvement, visible-response pedagogy/leakage checks, and thinking quality. Thinking-enabled experiments use DeepSeek-R1-0528-Qwen3-8B; the no-thinking baseline uses Qwen-2.5-7B. As in PedagogicalRL, the solve-rate test is conditioned on the completed dialogue, so it should not be read as an independent transfer test.
- **What is distinctive:** It explicitly rewards pedagogical planning inside the reasoning trace, not just the final tutoring response. Its five-way ablation reports the strongest combined results when theory-based prompting and thinking reward are used together.
- **Club use:** The Pólya sequence is a practical planning scaffold to try in prompts and tutor training. The club should evaluate visible behavior and independent transfer separately; its members do not need access to hidden reasoning to decide whether the tutor asks useful questions, adapts its hints, and avoids giving away the answer.
- **Primary source:** [arXiv paper, including prompts and training details](https://arxiv.org/abs/2601.14560).

### 3. EduAlign — *Cultivating Helpful, Personalized, and Creative AI Tutors: A Framework for Pedagogical Alignment using Reinforcement Learning* / *EduAlign: Aligning Educational LLMs for Helpfulness, Personalization, and Creativity* (arXiv:2507.20335)

> **What you need to know (ELI5):** EduAlign trains tutors against separate goals for helpfulness, personalization, and creativity; its latest version uses IdealPO, while the earlier version used GRPO. The three goals make a practical review rubric for club tutors, even without reproducing the training system.

- **Problem:** General-purpose assistants may be informative without being learner-centered, personalized, or creativity-supporting; a single reward score can hide conflicts among these goals.
- **Technology and solution:** The 2025 v1 builds an HPC reward model (Helpfulness, Personalization, Creativity) from 8,000 educational Q&A examples with expert and model annotation, then uses weighted multi-objective reward scoring and GRPO on 2,000 prompts. The latest v2, revised Oct. 4, 2026, retains the three objectives but uses iterative rubric-guided synthesis, an Edu-RM reward model, and a new multi-objective algorithm called Ideal-point Policy Optimization (IdealPO), tested with 7B and 32B models.
- **What is distinctive:** It treats responsible guidance, learner-specific adaptation, and creative exploration as separate measurable dimensions. Version history matters: **GRPO describes v1; it is not the optimization algorithm in current v2.**
- **Club use:** Borrow the three dimensions as a human review rubric: Was the answer useful and responsible? Did it fit the learner’s level and interests? Did it invite exploration? Make the club’s own examples first; neither version proves that the reward model’s scores are a substitute for club-member feedback.
- **Primary sources:** [latest arXiv version, v2](https://arxiv.org/abs/2507.20335); [original v1 matching the supplied title and GRPO summary](https://arxiv.org/abs/2507.20335v1).

### 4. Scarlatos et al. — *Training LLM-based Tutors to Improve Student Learning Outcomes in Dialogues* (arXiv:2503.06424)

> **What you need to know (ELI5):** The training setup favors tutor turns that are pedagogically sound and likely to let the student answer the next question correctly. That makes it a useful model for judging the next hint, but it does not directly measure later test scores or independent learning.

- **Problem:** A tutor can sound pedagogical while choosing a task or hint that is too difficult for a student to answer.
- **Technology and solution:** The authors generate candidate tutor turns from human MathDial tutors and LLMs. LLMKT, a knowledge-tracing student model, predicts the chance of a correct **next student response**; GPT-4o scores tutor turns against a rubric for accuracy, progress, mistake identification, strategic hinting, and withholding answers. The combined scores make preference pairs for distillation followed by DPO fine-tuning of Llama 3.1 8B.
- **What is distinctive:** It cheaply trains a smaller open model by balancing an explicit estimate of student success with a separate pedagogical rubric.
- **Club use:** The design suggests evaluating whether a hint makes the *next step* answerable while preserving longer-term learner agency. A club can manually judge this at first, instead of trusting an LLM simulator as ground truth.
- **Accuracy correction:** Despite the title’s “learning outcomes” phrasing, this method is not RL over complete simulated dialogues and does not optimize a downstream post-test score. It uses a predictor of the student’s next-turn correctness.
- **Primary source:** [arXiv paper and methodology](https://arxiv.org/abs/2503.06424); [authors’ code repository](https://github.com/umass-ml4ed/tutorbot-dpo).

### 5. SocraticLM — *SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models* (NeurIPS 2024)

> **What you need to know (ELI5):** SocraticLM uses multi-turn teaching dialogues and varied simulated learner states to train a model to ask questions and scaffold reasoning. Its dialogue patterns can help design practice scenarios and learner personas, though those simulations do not establish real-world learning gains.

- **Problem:** A question-answering tutor can give students explanations without involving them in the problem-solving process.
- **Technology and solution:** A “Dean–Teacher–Student” multi-agent pipeline creates SocraTeach, a dataset of 35,000 multi-round Socratic math dialogues (reported as equivalent to 208,000 single turns). Six simulated student cognitive states are used. The authors fine-tune on SocraTeach with three strategies to balance teaching and reasoning, and assess teaching across five pedagogical dimensions.
- **What is distinctive:** It combines a sizable multi-turn Socratic dialogue dataset, learner-state variation, and a purpose-built evaluation of “thought-provoking” teaching.
- **Club use:** Use its structure to prototype several learner personas and ask open-ended questions that get students to articulate their reasoning. Validate the personas and moves with actual club learners; simulated profiles alone do not establish personalization or learning gains.
- **Citation correction:** The supplied arXiv ID **2401.13275 is unrelated**: it is *Can AI Assistants Know What They Don’t Know?* The SocraticLM paper is documented in the official NeurIPS 2024 proceedings; I did not find an arXiv ID for it in the primary records checked.
- **Primary sources:** [official NeurIPS paper page](https://papers.nips.cc/paper_files/paper/2024/hash/9bae399d1f34b8650351c1bd3692aeae-Abstract-Conference.html); [authors’ dataset and code](https://github.com/Ljyustc/SocraticLM); [the unrelated arXiv ID](https://arxiv.org/abs/2401.13275).

### 6. MathTutorBench — *A Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors* (arXiv:2502.18940; EMNLP 2025)

> **What you need to know (ELI5):** A model that can solve math may still be poor at diagnosing a student’s thinking or giving a useful hint; MathTutorBench tests those tutor skills separately. Use its task structure to build a club-specific evaluation set, rather than relying on answer accuracy alone.

- **Problem:** Math accuracy and text similarity do not tell us whether a model understands student work, locates mistakes, scaffolds reasoning, or avoids answer leakage.
- **Technology and solution:** An open benchmark with seven tasks grouped into math expertise, student understanding, and teacher response generation. It includes problem solving, Socratic questioning, checking student solutions, locating/correcting mistakes, and generating scaffolded responses. A learned reward model scores open-ended pedagogical responses using preference comparisons between more and less effective tutoring utterances.
- **What is distinctive:** It separates solving ability from tutoring ability and makes open-ended scaffolding easier to compare across models. The paper reports a pedagogy/subject-expertise trade-off and more difficulty for models in longer dialogues.
- **Club use:** Use it as a reference for a small club-specific evaluation set. Keep separate scores for subject correctness, diagnosing a learner’s current understanding, and the quality of guidance; do not collapse these into one “tutor quality” number.
- **Primary sources:** [ACL Anthology / EMNLP 2025 proceedings](https://aclanthology.org/2025.emnlp-main.11/); [arXiv paper with task details](https://arxiv.org/abs/2502.18940); [official benchmark repository](https://github.com/eth-lre/mathtutorbench).

### 7. MathDial — *A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems* (arXiv:2305.14536; EMNLP 2023 Findings)

> **What you need to know (ELI5):** MathDial provides multi-turn tutoring conversations centered on student mistakes, showing how tutors can probe and scaffold without immediately revealing solutions. It is a useful source of dialogue patterns and annotation ideas for a club’s own subject materials.

- **Problem:** High-quality real tutoring dialogues are hard to collect at scale; ordinary problem-solving examples do not show how teachers respond to a learner’s errors.
- **Technology and solution:** Human teachers tutor LLM-simulated students prompted to make common mistakes. The resulting dataset contains about 3,000 one-to-one multi-turn math tutoring dialogues, with teacher-move annotations. The authors show it can be used to fine-tune models and evaluate the trade-off between helping students solve problems and revealing solutions.
- **What is distinctive:** The dialogues preserve actual teacher turns while using simulated students to produce varied, error-centered tutoring situations.
- **Club use:** Treat it as a source of dialogue patterns and annotation ideas, not a complete fit for your club. Build a small, consented set around your subject and recurring misconceptions; label each tutor turn by intent and whether it helped the learner make progress.
- **Primary source:** [arXiv paper](https://arxiv.org/abs/2305.14536); [authors’ code and dataset](https://github.com/eth-nlped/mathdial).

### 8. Intent Matters — *Intent Matters: Enhancing AI Tutoring with Fine-Grained Pedagogical Intent Annotation* (BEA 2025)

> **What you need to know (ELI5):** Labeling a tutor turn with its specific purpose—such as probing, guiding focus, or revealing an answer—makes teaching behavior easier to control and evaluate. A small version of this intent vocabulary could help the club specify what its tutor should do next.

- **Problem:** Broad turn labels can be too coarse for analysis or controllable response generation.
- **Technology and solution:** Re-annotates part of MathDial using a taxonomy of eleven pedagogical intents, via an automated annotation framework. The authors fine-tune an LLM with the fine-grained labels and compare it with models trained on MathDial’s original four labels; automatic and qualitative evaluations report improved pedagogical alignment.
- **What is distinctive:** It makes the tutor’s intended action more specific, enabling control and evaluation at the level of the move (rather than just a broad “good tutor response” label).
- **Club use:** Adapt a compact intent set for your club, such as diagnose, probe, hint, focus attention, validate reasoning, and explain. Let club mentors review a sample of labels before using them in prompts or training data.
- **Primary sources:** [ACL Anthology paper page and PDF](https://aclanthology.org/2025.bea-1.63/); [authors’ annotation code and data](https://github.com/Kpetyxova/autoTree/tree/main/mathdial).

### 9. LearnLM — *LearnLM: Improving Gemini for Learning* (arXiv:2412.16429)

> **What you need to know (ELI5):** LearnLM treats teaching as a behavior the model must be instructed and trained to follow, rather than assuming a generally helpful assistant will automatically teach well. Its scenario-based teaching instructions are a strong starting point for a prompt-first club prototype.

- **Problem:** Default conversational behavior presents information, while many learning settings require discovery, appropriately paced guidance, and nuanced instructions.
- **Technology and solution:** Frames tutoring as pedagogical instruction-following: conversations are conditioned on system instructions defining the desired teaching behavior. It combines supervised fine-tuning data with human preference collection and reward models for RLHF, and co-trains with Gemini’s broader post-training mixture. Its evaluation uses scenario-based multi-turn conversations role-played by educators and judged by pedagogy experts.
- **What is distinctive:** It makes pedagogical behavior configurable per task or learner scenario instead of requiring one fixed tutor persona, and evaluates full interactions with experts rather than only isolated answers.
- **Club use:** This is the most direct starting point for a prompt-based pilot: write system instructions for club goals (e.g., ask what the learner has tried, offer one hint at a time, adapt after errors, check understanding before moving on) and assess them in realistic club scenarios. The paper describes large-scale Gemini co-training, not a lightweight recipe for independently reproducing the model.
- **Primary source:** [arXiv paper](https://arxiv.org/abs/2412.16429).

### Established intelligent tutoring findings

#### 10. Bloom — *The 2 Sigma Problem: The Search for Methods of Group Instruction as Effective as One-to-One Tutoring* (1984)

> **What you need to know (ELI5):** Bloom’s famous studies found large gains for one-to-one tutoring and mastery learning compared with conventional classes, inspiring decades of tutoring research. For a club tutor, the actionable idea is to check understanding, give targeted correction, and revisit material—not to treat “two sigma” as a promised result.

- **Problem:** One-to-one tutoring appeared to produce learning outcomes that ordinary classrooms struggled to match. Bloom posed the practical question of how group instruction could approach effective tutoring.
- **Technology and solution:** This is a research synthesis and challenge paper, not an AI study. Bloom summarized two doctoral studies that randomly assigned students to conventional classes, classroom mastery learning with feedback and corrective work, or one-to-one/small-group tutoring. He reported average achievement differences of roughly two standard deviations for tutoring and one for mastery learning relative to conventional instruction.
- **What is distinctive:** It framed the “2 sigma problem” that motivated decades of tutoring research. The result is rooted in a small set of early studies and should be treated as an influential reference point, not a universal modern effect size or a performance target an AI can claim to match.
- **Club use:** Borrow the mastery-learning loop: check understanding, give targeted corrective support, and revisit the idea before moving on. Measure your own learning outcomes rather than translating Bloom’s reported effect to an AI tutor.
- **Primary source:** [Educational Researcher paper](https://doi.org/10.3102/0013189X013006004).

#### 11. Aleven & Koedinger — *An Effective Metacognitive Strategy: Learning by Doing and Explaining with a Computer-Based Cognitive Tutor* (2002)

> **What you need to know (ELI5):** Students using a Cognitive Tutor did better when prompted to explain their own problem-solving steps, including on transfer problems. A club tutor can use short “why did you take that step?” prompts to make learners do some of the thinking.

- **Problem:** Students can complete guided problem-solving practice without building a strong explanation of why their steps work. Self-explanation can promote deeper learning, but it needs practical support during real classroom work.
- **Technology and solution:** Two classroom experiments added prompts for students to explain their steps while practicing with a computer-based Cognitive Tutor. Students who explained their steps showed greater understanding, gave better explanations of their solution steps, and performed better on transfer problems than students who did not receive that support.
- **What is distinctive:** It tests a small, scalable instructional move—ask the learner to explain their own work—inside an established step-based ITS, rather than adding more explanation from the tutor.
- **Club use:** After a learner takes a step, ask what principle justifies it or how it connects to the goal. Use the reply to decide whether to advance, probe a misconception, or offer a hint; keep the request short enough that it supports learning rather than adding busywork.
- **Primary source:** [Cognitive Science paper](https://doi.org/10.1207/s15516709cog2602_1).

#### 12. VanLehn — *The Relative Effectiveness of Human Tutoring, Intelligent Tutoring Systems, and Other Tutoring Systems* (2011)

> **What you need to know (ELI5):** VanLehn’s review found that some step-based intelligent tutors performed comparably to human tutoring in the studies analyzed, challenging the idea that computers can only mark whole answers. It supports responding to a learner’s current step, while the results do not guarantee equivalence for every tutor or setting.

- **Problem:** Common accounts assumed tutoring became more effective as interaction became more fine-grained, with human tutoring far ahead of computer tutors. The evidence for that simple ranking was unclear.
- **Technology and solution:** This review compares experiments involving no tutoring, answer-based computer instruction, step-based or substep-based ITSs, and human tutors. It reports effect sizes of about *d* = 0.79 for human tutoring and *d* = 0.76 for ITSs against no-tutoring instruction—well below the often-repeated *d* = 2.0 estimate for human tutoring.
- **What is distinctive:** It separates tutoring systems by interaction granularity and challenges the idea that a human tutor necessarily has a dramatically larger effect than a step-based ITS. The estimates summarize a particular set of studies; the small difference is not proof of universal equivalence.
- **Club use:** Design the tutor to respond to the learner’s current step, not only mark a whole answer right or wrong. Compare that design with a simple answer-and-explanation baseline in your own pilot.
- **Primary source:** [Educational Psychologist review](https://doi.org/10.1080/00461520.2011.611369).

#### 13. Ma et al. — *Intelligent Tutoring Systems and Learning Outcomes: A Meta-Analysis* (2014)

> **What you need to know (ELI5):** Across many studies, intelligent tutoring systems outperformed several common alternatives, while the analysis found no statistically significant difference from individualized human tutoring or small-group teaching. This supports testing adaptive feedback in the club, with results compared against the club’s actual alternative.

- **Problem:** Results from individual ITS studies varied across subjects, systems, outcomes, and comparison groups, making it difficult to estimate the broader evidence.
- **Technology and solution:** A meta-analysis synthesized 107 effect sizes from 14,321 participants. ITSs showed higher achievement than teacher-led large-group instruction (*g* = 0.42), non-ITS computer-based instruction (*g* = 0.57), and textbooks/workbooks (*g* = 0.35). The analysis found no statistically significant difference from individualized human tutoring (*g* = −0.11) or small-group instruction (*g* = 0.05).
- **What is distinctive:** It gives a broad evidence baseline across educational levels and subject domains, including comparison types. “No significant difference” is not a formal equivalence finding, and these pooled results do not promise the same gains for a new tutor or setting.
- **Club use:** Treat adaptive feedback as a research-backed design direction, while measuring outcomes against the club’s actual alternative (such as mentor help or self-study) and subject matter.
- **Primary source:** [Journal of Educational Psychology meta-analysis](https://doi.org/10.1037/a0037123).

### LLM tutoring alignment, evaluation, and field evidence

#### 14. Sonkar et al. — *Pedagogical Alignment of Large Language Models* (Findings of EMNLP 2024)

> **What you need to know (ELI5):** Training on synthetic preference pairs for “helpful hint” versus “give the answer” improved models’ pedagogical-alignment scores over supervised fine-tuning. The paper offers a preference-training recipe for later experiments, but its scores are not measured student learning gains.

- **Problem:** General LLMs often give the answer immediately instead of guiding a learner, while high-quality pedagogical preference data is scarce.
- **Technology and solution:** The authors create synthetic preference pairs from the structured CLASS tutoring framework and train Llama, Mistral, and Zephyr models with Direct Preference Optimization (DPO), Identity Preference Optimization (IPO), and Kahneman-Tversky Optimization (KTO), comparing them with supervised fine-tuning. They also introduce perplexity-based measures of the model’s preference for hints/guidance versus direct solutions.
- **What is distinctive:** It makes pedagogical alignment a preference-learning objective and tests consistency across eight conversation rounds. The paper reports alignment-accuracy improvements over SFT (including 13.1% for Llama and 8.7% for Mistral); these are model/task metrics, not measured student learning gains. Although the method is called “learning from human preferences,” its preference pairs are generated synthetically from CLASS prompts rather than directly collected human rankings.
- **Club use:** If the club later fine-tunes a model, compare mentor-approved preference pairs (a useful hint versus an answer leak) against ordinary supervised examples. Start with club-reviewed labels and learner outcomes; synthetic preference scores alone are not evidence of teaching effectiveness.
- **Primary source:** [Findings of EMNLP 2024 paper and PDF](https://aclanthology.org/2024.findings-emnlp.797/).

#### 15. Maurya et al. — *Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors* (NAACL 2025)

> **What you need to know (ELI5):** MRBench scores tutor replies across eight dimensions, including finding a student’s mistake, guiding them, and avoiding answer revelation. Its rubric can help the club evaluate feedback consistently, especially when students are stuck.

- **Problem:** Evaluations of tutor responses used inconsistent rubrics, which made comparisons difficult and often failed to distinguish a correct answer from a useful teaching move.
- **Technology and solution:** The authors introduce MRBench, a math mistake-remediation benchmark with 192 dialogue contexts and 1,596 responses from seven LLM-based and human tutors. Human annotators label eight dimensions: mistake identification, mistake location, answer revelation, guidance, actionability, coherence, tutor tone, and human-likeness. The paper also checks Prometheus2 and Llama-3.1-8B as automated evaluators against human judgments.
- **What is distinctive:** MRBench supplies both a reusable human-annotated comparison set and a concrete multi-part tutoring rubric. It complements MathTutorBench (entry 6), which covers broader open-ended tutoring tasks; MRBench focuses specifically on responses to math mistakes and confusion.
- **Club use:** Adapt the rubric to your subject, especially “did it locate the learner’s error?”, “is the next action clear?”, and “did it reveal too much?”. Human-review a sample before treating model-judge ratings as trustworthy.
- **Primary source:** [NAACL 2025 paper and benchmark details](https://aclanthology.org/2025.naacl-long.57/); [authors’ repository](https://github.com/kaushal0494/UnifyingAITutorEvaluation).

#### 16. Cohn et al. — *A Theory of Adaptive Scaffolding for LLM-Based Pedagogical Agents* (AAAI 2026; arXiv:2508.01503)

> **What you need to know (ELI5):** This work connects evidence about what a learner knows to a decision about which hint or scaffold to offer next. It is a close design reference for tracking observable attempts and errors to adapt club tutoring, though it does not establish causal learning gains.

- **Problem:** Open-ended LLM agents can converse flexibly, but may lack an explicit account of what evidence shows a learner’s current understanding and how that evidence should change the next intervention.
- **Technology and solution:** The framework combines Evidence-Centered Design (ECD) for assessment evidence, Social Cognitive Theory (SCT), and the Zone of Proximal Development (ZPD) to guide adaptive support. Inquizzitor implements it as a GPT-4o formative-assessment agent with a scoring/evidence module and a decision module for feedback. The study involved 104 sixth-grade students in a three-week middle-school Earth Science curriculum, with 282 assessments and 288 agent conversations; the authors evaluate rubric-scoring agreement and whether feedback reflects intended constructs.
- **What is distinctive:** It connects an explicit evidence store and rubric-based formative assessment to the next scaffold, updating learner evidence after each interaction and reducing scaffolding as evidence of mastery grows. This is a theory-grounded design and classroom implementation study, not a randomized demonstration of learning gains.
- **Club use:** Represent learner state with observable evidence (attempts, explanations, recurring errors) and map it to a small set of next actions. This is a useful design pattern for an adaptive controller; avoid assigning fixed “learning styles” or inferring abilities from sparse chat.
- **Primary sources:** [AAAI 2026 proceedings paper](https://ojs.aaai.org/index.php/AAAI/article/view/37154); [2025 arXiv preprint](https://arxiv.org/abs/2508.01503); [authors’ appendices and code](https://github.com/claytoncohn/AAAI26).

#### 17. Kestin et al. — *AI tutoring outperforms in-class active learning: an RCT introducing a novel research-based design in an authentic educational setting* (Scientific Reports, 2025)

> **What you need to know (ELI5):** In a short college-physics study, students using a custom GPT-4 tutoring package learned more on immediate tests in less time than students in the active-learning comparison. It is a useful blueprint for a structured pilot, not evidence that a generic chatbot beats teaching across subjects or over the long term.

- **Problem:** It was unclear whether a generative AI tutor deliberately designed around learning science could support measurable learning as well as or better than a strong active-learning class.
- **Technology and solution:** In a randomized study of 194 eligible Harvard introductory-physics students, the authors compared a custom GPT-4 tutor—with carefully written, question-specific prompts, structured scaffolding, and instructional videos—with in-class active-learning lessons on the same content. Students in the AI condition scored higher on immediate post-tests in less time and reported higher engagement and motivation.
- **What is distinctive:** This is a learning-outcome study of a designed tutoring experience, not just a benchmark of model responses. The intervention was a package (custom prompts, scaffolding, videos, and self-paced use), covered two physics lessons at one university, and did not test delayed retention; it does not establish that a generic chatbot will outperform classroom teaching.
- **Club use:** Treat it as evidence that content design, pacing, and pedagogy can make an AI study experience useful. Prototype one short, structured unit and compare pre/post or fresh-problem performance, time, and participation with your club’s normal format.
- **Primary source:** [Scientific Reports paper and full text](https://www.nature.com/articles/s41598-025-97652-6); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/40537565/).

#### 18. Oreopoulos & Low — *One Click Away: AI Tutoring with Khanmigo in a Two-Year School Experiment* (NBER Working Paper 35620, August 2026)

> **What you need to know (ELI5):** The two-year school experiment found modest math gains from assigning Khanmigo, similar to non-AI practice, while most students used the tutor infrequently. For the club, this makes ease of use and meaningful help-seeking part of the tutoring design—not just model quality.

- **Problem:** A coaching-style AI tutor may be pedagogically well designed yet have little effect if students rarely use it in a meaningful way.
- **Technology and solution:** In a two-year cluster-randomized experiment across 18 Tennessee middle schools, Khanmigo was configured to coach students during existing remedial math practice rather than simply provide answers. Assignment to the Khanmigo condition raised achievement by about 0.06–0.08 standard deviations over a school year; the authors report gains similar in size to Khan Academy practice without AI. Ninety-six percent tried Khanmigo at least once, but the median student messaged it on only about one-third of practice days and used it in just 17% of exercise sessions with a mistake.
- **What is distinctive:** It foregrounds uptake and interaction quality as implementation constraints: giving students access did not lead most of them to engage in substantive tutoring dialogue. The paper is a 2026 working paper, and the estimated effect of a full year of active participation is an extrapolation, not the randomized assignment effect.
- **Club use:** Design the tutor’s entry point and feedback loop so students know when to ask for help and can use it with low friction. Track meaningful help-seeking and completion, not just accounts created or whether learners opened the tool.
- **Primary source:** [NBER Working Paper 35620](https://www.nber.org/papers/w35620).

#### 19. Wang et al. — *Tutor CoPilot: A Human-AI Approach for Scaling Real-Time Expertise* (arXiv:2410.03017, revised 2025)

> **What you need to know (ELI5):** Tutor CoPilot gives human tutors suggestions during live sessions, and the trial found improved short-term topic mastery, especially for students of lower-rated tutors. A club with peer mentors could adapt this human-in-the-loop approach instead of making the AI tutor students by itself.

- **Problem:** Novice tutors may lack the contextual expertise to choose a productive next move during a live session, while fully automated tutors cannot draw on a human tutor’s knowledge of the student and curriculum.
- **Technology and solution:** Tutor CoPilot is a tutor-facing assistant that uses the Bridge method to turn experienced educators’ think-aloud reasoning into strategy-conditioned suggestions. It uses the current lesson and recent conversation, offers multiple possible pedagogical moves, and lets a human tutor choose, edit, or ignore them. In a preregistered RCT involving about 900 tutors and 1,800 K–12 students, access increased session exit-ticket mastery by 4 percentage points overall and 9 points for students of lower-rated tutors.
- **What is distinctive:** It augments a human rather than replacing the tutor, preserves tutor agency, and tests the system in live sessions. The reported gains are proximal exit-ticket results; the study did not find a statistically significant improvement on end-of-year math tests, and some suggestions were not grade-level appropriate.
- **Club use:** A strong option if your club has peer mentors: show the mentor a suggested question, hint, or analogy and let them decide whether it fits the learner. Review suggestions for subject fit and age/level before deployment.
- **Primary sources:** [arXiv paper and current version](https://arxiv.org/abs/2410.03017); [authors’ implementation repository](https://github.com/rosewang2008/tutor-copilot).

#### 20. Shi et al. — *From Solver to Tutor: Evaluating the Pedagogical Intelligence of LLMs with KMP-Bench* (AAAI 2026; arXiv:2603.02775)

> **What you need to know (ELI5):** KMP-Bench tests both whole tutoring dialogues and specific K–8 math teaching skills, showing why solving a problem is not the same as teaching it. Use its two-level evaluation idea for club materials; benchmark gains alone do not show improved student learning.

- **Problem:** Standard math accuracy tests do not measure whether a model can sustain multi-turn teaching, detect and explain errors, or create appropriate practice.
- **Technology and solution:** KMP-Bench evaluates K–8 math tutoring in two parts: KMP-Dialogue scores responses against six principles (Challenge, Explanation, Modelling, Practice, Questioning, Feedback), while KMP-Skills tests multi-turn problem solving, error detection/correction, and problem generation. The authors use LLM-generated dialogue components, human verification for the 4.6K-dialogue evaluation set, and a separate 150K-dialogue training corpus, KMP-Pile. Fine-tuning a 7B model on KMP-Pile improves its benchmark results over its base model.
- **What is distinctive:** It combines holistic dialogue evaluation with granular tutoring skills and provides large-scale pedagogical training data. Its benchmark-generation and scoring pipeline is partly model-generated, so the gains are benchmark evidence, not yet evidence of improved learning with students.
- **Club use:** Borrow the two-level evaluation structure: test complete interactions and specific skills separately. Build a smaller set from actual club material, and manually inspect generated cases and model-judge scores before relying on them.
- **Primary sources:** [AAAI 2026 proceedings paper](https://ojs.aaai.org/index.php/AAAI/article/view/40578); [arXiv paper and full text](https://arxiv.org/abs/2603.02775).

### Additional evidence and design papers

#### 21. Bastani et al. — *Generative AI without guardrails can harm learning: Evidence from high school mathematics* (PNAS, 2025)

> **What you need to know (ELI5):** Giving students a general-purpose GPT-4 tool improved their assisted practice but left them worse on a later unaided math exam; the teacher-configured tutoring version avoided that harm. This is a strong reason to measure what learners can do after the AI is gone, not just while they use it.

- **Problem:** Students may get more practice done with an AI assistant while learning less independently, especially if the assistant supplies solutions too readily.
- **Technology and solution:** A randomized study in one Turkish high school compared GPT-4 access, a teacher-guardrailed GPT Tutor, and no AI across roughly 1,000 students. The tutor condition used teacher-provided correct solutions and common mistakes, plus instructions to give hints rather than answers. Students with the unguarded GPT-4 scored 17% lower than controls on the unassisted exam; the guarded tutor avoided that harm but did not exceed the control group.
- **What is distinctive:** It directly tests for a gap between supported practice and unaided performance. The result is consequential but bounded to one school, mathematics, and a near-term exam.
- **Club use:** Set a no-AI follow-up task alongside any tutoring pilot. Compare independent performance on a fresh but related problem, and use mentor-reviewed misconception notes and hint-first rules as guardrails.
- **Primary sources:** [PNAS article and DOI](https://doi.org/10.1073/pnas.2422633122); [full text in PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC12232635/).

#### 22. Kraft, Schueler & Falken — *What Impacts Should We Expect from Tutoring at Scale? Exploring Meta-Analytic Generalizability* (Review of Educational Research, 2026)

> **What you need to know (ELI5):** Tutoring tends to have smaller effects when delivered to many students under ordinary large-scale conditions than in tightly supported studies. Plan for real club attendance, tutor-to-student ratios, and session frequency when estimating impact.

- **Problem:** Small, closely managed tutoring studies can overstate what a program will achieve when it expands and serves students with varied attendance and support.
- **Technology and solution:** The authors reanalyze 263 randomized controlled trials of preK–12 tutoring and examine how effects vary with scale, dosage, tutor-student ratio, and implementation. The estimated impact is about 0.22 standard deviations for programs serving 400–999 students and 0.16 SD for those serving 1,000 or more, compared with about 0.40 SD across the full sample.
- **What is distinctive:** It centers generalizability and implementation conditions in the evidence synthesis, rather than treating an average tutoring effect as portable to every program. It is not specific to AI or student clubs.
- **Club use:** Record who actually attends, how often they use the tutor, and how much human support is available. Treat outcomes from an intensively supported pilot as a best-case estimate until use is tested under normal club conditions.
- **Primary source:** [Review of Educational Research article](https://journals.sagepub.com/doi/10.3102/00346543261446660).

#### 23. Corbett & Anderson — *Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge* (User Modeling and User-Adapted Interaction, 1995)

> **What you need to know (ELI5):** Knowledge tracing estimates how likely a learner is to have mastered each skill from their sequence of right and wrong attempts. It is a practical model for a tutor that remembers demonstrated skills instead of relying on a vague overall learner profile.

- **Problem:** A tutor needs to decide what to practice next without assuming that one correct answer means a skill is mastered or that one error means it is unknown.
- **Technology and solution:** Bayesian Knowledge Tracing models each production-rule skill with probabilities for initial mastery, learning, guessing, and slipping, and updates its mastery estimate from observed answers. The original implementation was used in the Cognitive Tutor for programming to select exercises and predict performance.
- **What is distinctive:** It gives a simple, interpretable learner-state model based on repeated evidence, with a separate estimate for each skill. It expects attempts to be tagged to skills and does not by itself interpret open-ended chat or explanations.
- **Club use:** Start with a small skill map and update mastery only when a learner solves or explains relevant tasks. Keep the evidence visible to mentors and allow correction when the model misreads an attempt.
- **Primary source:** [Springer journal article](https://link.springer.com/article/10.1007/BF01099821).

#### 24. Alam et al. — *Does Knowing When Help Is Needed Improve Subgoal Hint Performance in an Intelligent Data-Driven Logic Tutor?* (AAAI 2023)

> **What you need to know (ELI5):** A logic tutor that noticed when a learner was struggling and offered a subgoal hint at that moment helped a low-prior-proficiency subgroup produce shorter proofs on a final test. It suggests that deciding *when* to intervene can matter as much as writing a good hint.

- **Problem:** Learners may not ask for help when they need it, while unsolicited hints can interrupt learners who are already making progress.
- **Technology and solution:** The discrete-math tutor predicts when a proof step is inefficient or help is likely needed, then proactively presents a data-driven subgoal hint. Students could also request hints on demand. In a study with 149 undergraduates (140 completing), the adaptive condition produced significantly shorter final-test proofs for the low-prior-proficiency subgroup; effects across the full sample were not significant.
- **What is distinctive:** It tests timing of intervention in a structured tutor and reports a subgroup benefit rather than claiming a universal advantage. It is a small, domain-specific study.
- **Club use:** First log observable signals such as repeated failed attempts, long stalls, or explicit confusion. Let students dismiss a proactive hint and compare adaptive prompts with an on-demand-only option.
- **Primary source:** [AAAI proceedings paper](https://ojs.aaai.org/index.php/AAAI/article/view/26887).

#### 25. Atkinson, Renkl & Merrill — *Transitioning From Studying Examples to Solving Problems: Effects of Self-Explanation Prompts and Fading Worked-Out Steps* (Journal of Educational Psychology, 2003)

> **What you need to know (ELI5):** Learners transferred knowledge better when worked examples gradually removed steps and prompted them to explain the underlying principle. A tutor can use this as a progression from showing, to partially completing, to asking the student to solve.

- **Problem:** Students can follow a worked solution without learning to choose and apply the underlying method on their own.
- **Technology and solution:** Across two experiments on probability tasks, the authors combined faded worked examples (successively omitting solution steps) with prompts asking learners to explain the principle behind a step. The combined approach improved near and far transfer without increasing learning time.
- **What is distinctive:** It is a controlled learning-science result about how to move from example study to independent problem solving, rather than a technology-specific tutoring method. Its tasks were structured probability problems.
- **Club use:** Have the tutor show one example, hide selected steps in the next, and then give the learner a similar fresh problem. Ask for a short explanation of why a step works instead of only asking for the answer.
- **Primary source:** [APA journal article](https://doi.org/10.1037/0022-0663.95.4.774).

#### 26. Puech et al. — *Towards the Pedagogical Steering of Large Language Models for Tutoring: A Case Study with Modeling Productive Failure* (Findings of ACL 2025)

> **What you need to know (ELI5):** StratL steers a tutor through a planned sequence of teaching moves using a transition graph, including a “try first, explain after” Productive Failure approach. It is a lighter-weight alternative to reinforcement learning when a club wants predictable multi-turn behavior.

- **Problem:** Ordinary prompts can make a tutor drift from a planned teaching approach over a multi-turn lesson.
- **Technology and solution:** The authors use StratL prompt optimization to encode a multi-turn pedagogical plan as a state-transition graph and steer the LLM through the planned stages. Their case study uses Productive Failure, where learners first try a challenging problem before receiving an explanation. A small field study with 17 students in Singapore examines plan-following.
- **What is distinctive:** It represents pedagogy as an explicit dialogue policy that can be inspected and adjusted, rather than learning the policy through RL. The field study assesses feasibility and plan adherence, not causal learning gains.
- **Club use:** Map a club activity into states such as “attempt,” “diagnose,” “hint,” “explain,” and “check transfer.” Prototype and revise this flow with mentors before collecting training data.
- **Primary source:** [ACL Anthology paper](https://aclanthology.org/2025.findings-acl.1348/).

#### 27. Chung et al. — *Effective Personalized AI Tutors via LLM-Guided Reinforcement Learning* (arXiv:2608.16907, 2026 preprint)

> **What you need to know (ELI5):** In a five-month Python course, students did better when the tutor adapted problem difficulty using evidence from their coding attempts and chats instead of following a fixed sequence. This is the clearest recent evidence here for personalizing the learning path, not just the wording of a hint.

- **Problem:** Fixed practice sequences may be too easy for some students and too difficult for others, while simple correctness-only rules miss useful evidence in code attempts and student conversations.
- **Technology and solution:** The system models tutoring as a partially observable decision process. It extracts richer learner signals from code attempts and chat, updates learner-state beliefs with particle filtering, and uses model-predictive control to select problem difficulty. A randomized study in ten Taipei high schools enrolled 1,047 students; the preregistered analysis included 770. The adaptive sequencing group scored about 0.15 SD higher on an unassisted exam than the fixed-sequence group, with larger gains reported for beginners.
- **What is distinctive:** The intervention randomizes adaptive versus fixed problem sequencing while keeping the chatbot and materials similar, helping isolate the value of adaptation. It is a 2026 preprint in a specific Python course and education context; the title’s “reinforcement learning” should not be read as response-model fine-tuning.
- **Club use:** Adapt the sequence of exercises based on demonstrated skill and engagement, while keeping tutor language and content comparable in a pilot. Use mentor-reviewed level changes and an unaided assessment to check whether adaptation helps.
- **Primary source:** [arXiv paper and full text](https://arxiv.org/abs/2608.16907).

#### 28. Worden et al. — *A Large Scale Randomized Control Trial Showing LLM Generated Feedback Helps Low-Knowledge Middle School Math Students with Short-Term Learning* (Learning @ Scale, 2026)

> **What you need to know (ELI5):** On thousands of middle-school math questions, expert-reviewed AI feedback helped students correct mistakes and succeed on the next problem, especially those with lower prior knowledge. The results support specific, timely feedback, but do not yet show long-term transfer.

- **Problem:** Correct/incorrect marks do not explain a misconception or tell a learner what to try next, and writing individualized feedback for every wrong answer is costly.
- **Technology and solution:** Two randomized studies tested feedback generated by Qwen-235B for common wrong answers across 572 problems. Educators reviewed the feedback, with an automated LLM check in the production process. In Study 1 (20,706 students), students were 16% more likely to correct the current answer and 7% more likely to answer the next problem correctly than with correctness-only feedback. Study 2 (12,950 students) found short-term outcomes comparable to teacher-written feedback.
- **What is distinctive:** It combines large-scale outcome evidence with an operational method for authoring, validating, and caching misconception-specific feedback; reported cost was about 5–10 cents per cached response. Outcomes are short-term and item-level, not evidence of sustained dialogue or durable learning.
- **Club use:** Build mentor-approved feedback for the most common mistakes in your materials, then let the model personalize its wording within those reviewed explanations. Track whether learners correct the misconception on a later, different problem.
- **Primary source:** [ACM proceedings paper](https://doi.org/10.1145/3774398.3811594).

#### 29. Oreopoulos et al. — *Making AI Tutoring Productive: Evidence from a Mastery-Based Math Practice Experiment* (NBER Working Paper 35621, 2026)

> **What you need to know (ELI5):** AI feedback helped students recover from wrong answers, with the clearest delayed-test gains when they had to reach mastery through repeated practice. Pair tutoring with retry and review loops instead of treating each response as a one-off chat.

- **Problem:** Helpful explanations may not lead to learning if learners move on before trying again or demonstrating mastery.
- **Technology and solution:** In a randomized experiment with more than 6,000 middle-school students in Hamilton County, students used NUMI AI tutoring or computer-assisted practice, with mastery-based or non-mastery practice workflows. The one-week delayed assessment showed the clearest gains when AI feedback was embedded in a mastery workflow; effects were marginal and largely limited to practiced items. AI users also made more accurate next attempts and needed fewer attempts or less time to correct errors.
- **What is distinctive:** It separately manipulates AI feedback and mastery-based practice, foregrounding the learning workflow around the model. It is a working paper with a short delayed test in one local setting.
- **Club use:** For a topic, have students retry after feedback and revisit the skill until they meet a transparent mastery criterion; include a later, changed problem to test transfer beyond the practiced item.
- **Primary source:** [NBER Working Paper 35621](https://www.nber.org/papers/w35621).

#### 30. LearnLM Team & Eedi — *AI tutoring can safely and effectively support students: An exploratory RCT in UK classrooms* (arXiv:2512.23633, 2025 preprint)

> **What you need to know (ELI5):** Expert-supervised LearnLM tutoring produced short-term misconception correction and transfer results comparable to human tutoring in a small classroom trial. It is a useful human-review model for a club pilot, but the study is too small to establish a clear learning advantage.

- **Problem:** Schools need to know whether AI-generated tutor messages can be safe and useful when used with real students, and how much expert review is required.
- **Technology and solution:** In five UK secondary schools, 165 students received tutoring in which LearnLM drafted messages under human expert supervision; experts approved or edited messages. The study reports immediate misconception correction comparable to human tutoring. For next-topic transfer, the supervised LearnLM group scored 66.2% versus 60.7% for the human-only condition; the 95% credible interval for the difference included zero.
- **What is distinctive:** It puts expert oversight directly into the classroom workflow and evaluates student responses rather than only model ratings. It is an exploratory, small-sample preprint, and the transfer estimate remains uncertain.
- **Club use:** Start with the tutor drafting a hint for a mentor to approve, edit, or reject. Log the edit reasons; those reviews can shape safer instructions and future examples.
- **Primary source:** [arXiv paper and full text](https://arxiv.org/abs/2512.23633).

#### 31. De Simone et al. — *From Chalkboards to Chatbots: Evaluating the Impact of Generative AI on Learning Outcomes in Nigeria* (World Bank Policy Research Working Paper 11125, 2025)

> **What you need to know (ELI5):** Students in a six-week after-school English program improved when they used Copilot with a trained teacher and structured prompts. This is evidence for a facilitated club-style format, while the study cannot separate the AI’s effect from extra instruction and teacher support.

- **Problem:** Evidence about generative AI in real learning programs is limited, especially in settings with constrained access to teachers and learning materials.
- **Technology and solution:** A randomized after-school program involved 1,328 volunteer students and used Microsoft Copilot/GPT-4 for English practice, paired student work, teacher facilitation, and starter prompts. Among the 759 students who completed the final assessment, the authors report gains of 0.23 SD in English and 0.31 SD on a composite measure on an immediate paper-and-pencil assessment.
- **What is distinctive:** It tests a real facilitated learning program rather than isolated model responses. Differential attrition (64% completion in treatment versus 50% in control), the bundled extra instruction, and the working-paper status limit causal interpretation of the AI component alone.
- **Club use:** The setup is relevant if your club can pair the tutor with a facilitator and collaborative tasks. Preserve the teacher/mentor role and assess both participation and independent performance.
- **Primary source:** [World Bank working paper PDF](https://documents1.worldbank.org/curated/en/099548105192529324/pdf/IDU-c09f40d8-9ff8-42dc-b315-591157499be7.pdf).

#### 32. Borchers & Shou — *Can Large Language Models Match Tutoring System Adaptivity? A Benchmarking Study* (arXiv:2504.05570, 2025 preprint)

> **What you need to know (ELI5):** When researchers removed or changed learner context, even strong LLMs often failed to change their teaching move in the way an adaptive tutoring system would. Test whether your tutor actually responds differently to a learner’s history, not just whether it can repeat that history.

- **Problem:** A tutor may mention details from a conversation yet still give the same kind of instruction regardless of a learner’s knowledge or need.
- **Technology and solution:** The authors test Llama 3 8B/70B and GPT-4o on 75 real tutoring scenarios (1,350 instructional moves) while ablating contextual information. They compare the models’ changed moves with established ITS behavior using text representations and a validated tutor-training classifier.
- **What is distinctive:** It treats adaptation as a counterfactual test: does the tutor change its action when relevant learner evidence changes? The study uses a fixed benchmark of scenarios and does not measure learning outcomes.
- **Club use:** For each evaluation case, create two versions that differ in one relevant learner fact, such as mastery or a misconception. Check that the next tutor action changes appropriately and that unrelated changes do not alter it.
- **Primary source:** [arXiv paper](https://arxiv.org/abs/2504.05570).

#### 33. Li et al. — *LongTutor: Benchmarking Large Language Models for Long-term Personalized Tutoring* (ACL 2026)

> **What you need to know (ELI5):** LongTutor tests whether a model can use a learner’s history to retrieve relevant evidence, diagnose what they know, and choose what to teach next; models still struggle with diagnosis and action selection. Treat those as three separate capabilities when building memory and adaptation.

- **Problem:** A tutor that can answer the current turn may still fail to remember learning across sessions or use that history to personalize its next teaching move.
- **Technology and solution:** The benchmark uses expert-annotated real learning logs and evaluates history evidence retrieval, student knowledge-state diagnosis, and adaptive teaching-action selection.
- **What is distinctive:** It separates long-term memory from inference about knowledge and from the teaching decision. It is a benchmark built around math learning logs, not a causal study of learner outcomes.
- **Club use:** Store a short evidence record for each skill, then separately inspect whether the system retrieved the right evidence, inferred the learner’s state correctly, and selected an appropriate next activity.
- **Primary source:** [ACL Anthology paper](https://aclanthology.org/2026.acl-long.1371/).

#### 34. Scarlatos et al. — *Simulated Students in Tutoring Dialogues: Substance or Illusion?* (ACL 2026)

> **What you need to know (ELI5):** A simulated student can sound realistic without making the same choices or learning changes as a real student. This paper provides ways to test student simulators before trusting them to train or score a tutor.

- **Problem:** Many tutoring systems are trained or evaluated against simulated learners, but it is unclear whether those simulations reproduce real students’ language, behavior, and knowledge change.
- **Technology and solution:** The authors define separate measures of linguistic, behavioral, and cognitive fidelity and benchmark methods against real-world math tutoring dialogue. Prompted simulators perform poorly on these measures; supervised fine-tuning and preference optimization improve some aspects but remain limited.
- **What is distinctive:** It distinguishes sounding like a learner from behaving like one and from modeling how knowledge changes. Simulator fidelity still does not prove that a simulator predicts real educational outcomes.
- **Club use:** If using simulated students for scenario generation or tutor tuning, validate their errors and responses against mentor-written cases and real, consented student interactions. Do not use simulator reward as your only success measure.
- **Primary source:** [ACL Anthology paper](https://aclanthology.org/2026.acl-long.1960/).

#### 35. Srinivasa et al. — *TutorBench: A Benchmark To Assess Tutoring Capabilities Of Large Language Models* (arXiv:2510.02663, 2025 preprint)

> **What you need to know (ELI5):** TutorBench tests models on many high-school and AP teaching situations, including explanations, feedback, and active-learning hints; even strong models often fail its tutoring criteria. It can help benchmark broad tutor skills, but it does not show how real students learn.

- **Problem:** Standard academic benchmarks test whether a model can answer subject questions, not whether it can teach across a range of tutoring tasks.
- **Technology and solution:** TutorBench contains 1,490 human-curated samples across six STEM subjects, covering adaptive explanations, feedback, and active-learning hints. It uses per-example rubrics and an LLM judge calibrated on 250 samples with three human ratings each (reported F1 of 0.81 against majority judgments). The paper reports that the strongest evaluated models pass fewer than 60% of criteria in key guidance, diagnosis, and support dimensions.
- **What is distinctive:** It spans a broader set of subjects and tutoring behaviors than math-only benchmarks and exposes failure at criterion level. It is a preprint benchmark with no direct student outcome measure.
- **Club use:** Borrow the rubric format to score mentor- and learner-authored examples. Use scores to find failure types, then confirm important ratings with human review.
- **Primary source:** [arXiv paper](https://arxiv.org/abs/2510.02663).

#### 36. Yang et al. — *MMTutorBench: The First Multimodal Benchmark for AI Math Tutoring* (ACL 2026)

> **What you need to know (ELI5):** A tutor that reads photos of handwritten work has to understand both the math and the student’s visual steps, and current multimodal models still trail human tutors. This benchmark is useful if club members will send images rather than type their work.

- **Problem:** Text-only tutor tests miss errors introduced when models read diagrams, handwritten steps, and student work from images.
- **Technology and solution:** MMTutorBench contains 770 step-based math problems and evaluates multimodal models on insight discovery, operation formulation, and operation execution using task-specific rubrics across six dimensions. The authors evaluate 12 multimodal LLMs and report a substantial gap from human tutors; OCR errors can reduce tutoring quality, and few-shot examples offer limited improvement.
- **What is distinctive:** It evaluates tutoring from visual student work rather than only text prompts and final answers. It is math-focused and benchmark-based, not a classroom outcome study.
- **Club use:** If students share photos, include legible and ambiguous handwriting, diagrams, and partial work in the club test set. Require the tutor to restate what it thinks the student wrote before diagnosing an error.
- **Primary source:** [ACL Anthology paper](https://aclanthology.org/2026.acl-long.1068/).

#### 37. Zhao et al. — *SHAPE: Unifying Safety, Helpfulness and Pedagogy for Educational LLMs* (ACL 2026)

> **What you need to know (ELI5):** SHAPE tests whether an educational model can stay helpful and pedagogical when students pressure it to give answers or bypass its rules. Its knowledge-mastery graph and explicit gates are useful design ideas for handling “just tell me” requests.

- **Problem:** Safety rules can make a tutor unhelpful, while a tutor optimized only for helpfulness may give away answers or follow harmful educational requests.
- **Technology and solution:** SHAPE introduces 9,087 student-question pairs and a framework that uses a knowledge-mastery graph to represent prerequisites and mastery gaps. Explicit gates route the model between teaching and problem-solving behaviors. Evaluation tests safety in two pedagogical jailbreak settings while checking that helpfulness remains high.
- **What is distinctive:** It combines safety, helpfulness, and pedagogy with an explicit representation of prerequisite knowledge rather than treating them as independent generic chat qualities. Evidence is benchmark performance, not student learning results.
- **Club use:** Make “give a hint,” “explain after an attempt,” and “show a worked solution” distinct allowed modes, with clear conditions for moving between them. Include answer-pressure scenarios in the review set.
- **Primary source:** [ACL Anthology paper](https://aclanthology.org/2026.acl-long.529/).

#### 38. Zhao, Knežević & Käser — *Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks* (ACL 2026)

> **What you need to know (ELI5):** Students can pressure a tutor in several ways to reveal a solution, so testing only ordinary “please help” prompts misses important failures. This paper offers an adversarial evaluation approach for answer-leakage defenses.

- **Problem:** A tutor may withhold a solution in a normal interaction but reveal it after persuasive or manipulative follow-up messages.
- **Technology and solution:** The authors test six adversarial student strategies and find common prompt-only simulated attackers weak at reproducing robust answer-seeking behavior. They fine-tune an attacker to probe the tutor’s answer-leakage defenses and propose the resulting setup as a standardized robustness benchmark.
- **What is distinctive:** It evaluates the tutor and the strength of the test attacker, showing that a weak student simulator can make a fragile tutor appear robust. The attack strategies remain proxies for real student behavior.
- **Club use:** Add escalating but age-appropriate pressure cases—deadline urgency, claims of already trying, requests for a full solution, and attempts to reframe answer-giving as a hint. Review both leakage and whether the tutor remains supportive.
- **Primary source:** [ACL Anthology paper](https://aclanthology.org/2026.acl-long.1412/).

#### 39. Petukhova & Kochmar — *Towards Reward Modeling for AI Tutors in Math Mistake Remediation* (LREC 2026)

> **What you need to know (ELI5):** A small reward model trained on carefully contrasted examples can learn which math-mistake feedback people prefer. It is a possible way to scale mentor feedback, but preference accuracy is not proof that students learn more.

- **Problem:** Fine-tuning a tutor to remediate mistakes requires examples that distinguish a useful explanation from a merely plausible response, while human preference labels are expensive.
- **Technology and solution:** The authors derive a hierarchy of feedback preferences from MRBench human comparisons, generate minimally contrastive synthetic response pairs, and train a Bradley–Terry preference model with a 0.5B backbone. The model reaches 0.69 pairwise accuracy using synthetic data and 0.74 when weighted and targeted synthetic data are combined.
- **What is distinctive:** It shows a route to a compact math-feedback reward model without collecting all preference pairs from humans, while anchoring the hierarchy in an existing human-judged benchmark. It predicts preferences, not learning outcomes.
- **Club use:** If mentor review becomes a bottleneck, create paired examples that differ in one teaching quality at a time (e.g., hint versus answer, targeted versus generic feedback). Keep human ratings in the evaluation set and check learner outcomes before using the reward model for training.
- **Primary source:** [LREC 2026 paper](https://aclanthology.org/2026.lrec-1.232/).

#### 40. Kobler et al. — *Your Students Don’t Use LLMs Like You Wish They Did: A Computational Analysis of Pedagogical Alignment of LLMs in Higher Education* (ACL 2026)

> **What you need to know (ELI5):** Real students often use LLMs to extract answers, and how the tool is embedded in a course strongly shapes that behavior. Design around the ways students will actually reach for the tutor, not only around idealized tutoring conversations.

- **Problem:** Tutor alignment evaluated through curated prompts may not reflect what students do when using an LLM for real coursework.
- **Technology and solution:** The authors analyze 12,650 messages from 500 conversations across four higher-education courses and validate six computational metrics of pedagogical alignment. They find answer extraction is common and that deployment context is a strong predictor of student behavior; optional use near deadlines and tools embedded in assignments can invite different patterns.
- **What is distinctive:** It studies naturally occurring interaction and the setting around the model, rather than relying only on benchmark prompts. It is observational higher-education research, so it does not establish that a particular design caused the observed behavior.
- **Club use:** Pilot the tutor in the real club workflow and review consented interaction patterns: what brings students to it, what they ask, and whether they act on guidance. Use that evidence to adjust the entry point and activity structure.
- **Primary source:** [ACL Anthology paper](https://aclanthology.org/2026.acl-long.875/).

**Deduplication note:** LearnLM (entry 9) and MathTutorBench (entry 6) were already covered in the first review and are not repeated. Entry 30 is a distinct classroom RCT of supervised LearnLM tutoring; it is not the LearnLM model report. MRBench (entry 15) and KMP-Bench (entry 20) are distinct evaluation resources. Entries 21–40 add evidence and methods not covered in the original set.

## Benchmarks and datasets inventory

**Scope and reading guide.** This is the most complete deduplicated inventory I could verify from the papers summarized here, primary-source searches, and Fab AI’s education benchmark program, checked through 2026-10-08. It covers resources that directly test tutor behavior, learner adaptation, pedagogical knowledge, student learning, or a close deployment concern; ordinary subject-answering sets appear only when a tutor study repurposes them. It includes peer-reviewed papers, preprints, and clearly labeled research/developer releases. The final column gives one short sentence on a paper or experiment and its result; “no causal outcome” means a benchmark score is not proof that students learned more.

**Fab AI source note.** I interpret “fab.ai” as Fab AI’s AI-for-Education program at [fab-ai.org](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/benchmarks/about), because the exact `fab.ai` address did not resolve in the research search. Fab AI says it mapped more than 6,000 education-AI papers across 11 quality areas; that mapping is a discovery index, not a benchmark itself, and its own benchmarks mostly test foundational model abilities rather than student learning outcomes.

### Tutor behavior, adaptation, and learning-outcome benchmarks

| Resource and primary source | What it contains / access | Paper or experiment result (one sentence max) |
|---|---|---|
| [MathTutorBench](https://aclanthology.org/2025.emnlp-main.11/) · [code/data](https://github.com/eth-lre/mathtutorbench) | Open math-tutor benchmark with seven tasks across math expertise, student understanding, and tutor response; includes expert/novice preferences and a learned response reward model. | The authors found solving skill did not ensure teaching skill, the reward model distinguished expert from novice responses, and performance became harder in longer dialogues. |
| [TutorBench](https://arxiv.org/abs/2510.02663) · [released data](https://huggingface.co/datasets/ScaleAI/TutorBench) | 1,490 expert-curated high-school/AP STEM cases for adaptive explanations, actionable feedback, and active-learning hints; paper and dataset release report slightly different row counts. | Evaluated models passed fewer than 60% of key guidance, diagnosis, and student-support criteria, with no model exceeding 56% overall under the paper’s main scoring. |
| [MRBench](https://aclanthology.org/2025.naacl-long.57/) · [V1–V3/data/code](https://github.com/kaushal0494/UnifyingAITutorEvaluation) | Mistake-remediation contexts from MathDial and Bridge, with 192 V1 contexts and 1,596 tutor responses annotated on eight dimensions; later releases add V2/V3 task data. | The original study found broad room for improvement across model tutor responses, and follow-on reward modeling reached 0.74 pairwise preference accuracy when combining weighted human-anchored and synthetic pairs. |
| [BEA 2025 shared task](https://sig-edu.org/sharedtask/2025) · [findings](https://aclanthology.org/2025.bea-1.77/) · [IndoML extension](https://kaushal0494.github.io/datathon/) | Shared-task splits use MathDial/Bridge conversations, human and seven-LLM tutor responses, and annotations for mistake identification and guidance; MRBench V3 has 300 contexts but only four dimensions for newly added examples. | Participating systems showed that mistake identification and response-quality assessment are feasible but remain imperfect, so the four-dimension extension should not be treated as equivalent to fully annotated MRBench V1. |
| [Developing an LLM Tutor Quality Evaluation Scale](https://aclanthology.org/2026.aimecon-sessions.15/) | Framework, not a standalone public test set: synthesizes 177 literature metrics into a 10-category, 41-item tutor-quality scale. | Exploratory factor analysis identified five constructs and a strong general factor, with confirmatory and human validation still needed. |
| [AITutor-EvalKit](https://aclanthology.org/2026.eacl-demo.32/) · [code/demo](https://github.com/kaushal0494/AITutor-EvalKit) | Open evaluation toolkit and UI for scoring mistake identification/location, guidance, and actionability on MRBench/BEA-style math dialogue responses. | Its Gemma-2-2B LoMTL evaluator reached 0.72 averaged accuracy and 0.60 macro-F1 on the held-out BEA set, versus 0.66/0.57 for GPT-5. |
| [GuideEval](https://arxiv.org/abs/2508.06583) | Authentic-dialogue-grounded tasks score three phases: infer learner state, orchestrate an instructional response, and elicit reflection; code/data availability should be checked before building on it. | Existing LLMs often failed to scaffold learners who were confused or needed redirection, while behavior-guided fine-tuning substantially improved guidance scores. |
| [SimulatorArena](https://aclanthology.org/2025.emnlp-main.1786/) | 909 annotated human–LLM conversations, including math tutoring, used to test whether LLM user simulators reproduce human behavior and judgments. | Profile-conditioned simulators reached Spearman ρ = 0.70 against human assistant judgments on math tutoring, supporting simulator use as a scalable proxy with validation caveats. |
| [LongTutor](https://aclanthology.org/2026.acl-long.1371/) · [code/data](https://github.com/liano3/LongTutor) | Real learning logs and interaction sequences from MOOCRadar and XES3G5M; separates history evidence retrieval, knowledge-state diagnosis, and teaching-action selection. | Models were comparatively stronger at retrieving relevant history than using it to diagnose mastery and choose an adaptive next action. |
| [KMP-Bench / KMP-Pile](https://arxiv.org/abs/2603.02775) · [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/40578) | K–8 math benchmark with KMP-Dialogue (about 4.6K evaluation dialogues) and KMP-Skills, plus a 150K-dialogue training corpus. | Leading models handled answer-verifiable tasks better than pedagogical judgment, while fine-tuning a 7B model on KMP-Pile improved its KMP-Bench scores. |
| [TutorMoments](https://github.com/allenai/tutormoments) · [preview data](https://huggingface.co/datasets/allenai/tutormoments-preview) · [report](https://tutormoments.allen.ai/static/paper/tutormoments-preview.pdf) | Moment-level benchmark drawn from 462 de-identified US tutoring sessions; the preview benchmark contains 520 scaffold-versus-rigor moments and model attempts. | Explicit prompts about when to help or hold back improved scores for all seven evaluated models, but the models still often over-help and the benchmark does not measure learning gains. |
| [EduClaw-Bench](https://arxiv.org/abs/2608.03206) | Long-horizon agent benchmark placing a simulated, knowledge-traced learner in a 30-day relationship with a tutor across 55 scenarios. | Across ten agent/model combinations, almost none sustained strong tutoring over the full horizon, although the authors report calibration and a small classroom validation. |
| [EduFair-Bench](https://arxiv.org/abs/2609.12949) | Controlled audit of five tutors across gender, immigration background, first language, and socioeconomic cues, using turn- and dialogue-level pedagogy metrics. | Model capability and fairness were largely unrelated, and pedagogy-specific RL redistributed demographic disparities rather than removing them. |
| [Borchers & Shou adaptivity benchmark](https://arxiv.org/abs/2504.05570) | 75 real tutoring situations and 1,350 response decisions test whether changing or removing learner context changes the model’s teaching move appropriately. | Most tested LLMs changed their instruction only marginally when relevant context changed, while GPT-4o sometimes became too directive. |
| [SHAPE](https://aclanthology.org/2026.acl-long.529/) | 9,087 student-question cases combine educational safety, helpfulness, pedagogy, and a knowledge-mastery graph for routing tutor behavior. | Its gating approach improved educational-jailbreak resistance while keeping helpfulness near the benchmark ceiling, but it did not test student learning. |
| [Adversarial answer-leakage benchmark](https://aclanthology.org/2026.acl-long.1412/) | Six simulated student pressure strategies test whether tutors reveal solutions after escalating persuasion. | Prompt-only attack students were weak probes, while a fine-tuned student attacker exposed more answer-leakage weaknesses. |
| [SafeTutors](https://arxiv.org/abs/2603.17373) · [code](https://github.com/RadiantCrystal/SafeTutors) | Multi-turn pedagogical-safety tests across three STEM subjects assess whether responses support or harm learning. | The paper reports substantial pedagogical harm across models, with harm rates rising sharply in multi-turn evaluations. |
| [EduGuardBench](https://arxiv.org/abs/2511.06890) | Evaluates teacher-role fidelity and adversarial safety across 14 models. | Incompetent tutoring was a common failure, and safety varied non-monotonically with model size, with some mid-sized models more vulnerable than larger ones. |
| [UniTutorBench](https://openreview.net/pdf?id=DKR4tgcW1I) | Preprint benchmark of 2,187 K–12 math cases spanning six tutoring tasks and ten fine-grained abilities, with step-level process measures. | Across 11 models, explanation performance did not predict scaffolding performance; Gemini-2.5-Flash fell 16% when moving from explanation to scaffolding. |
| [MMTutorBench](https://aclanthology.org/2026.acl-long.1068/) | 770 multimodal math problems with pedagogically important solution steps, scored on insight discovery, operation formulation, and execution. | Across 12 multimodal models, responses remained below human tutors, OCR pipelines reduced quality, and few-shot prompting helped little. |
| [MathExplain](https://arxiv.org/abs/2504.03197) | 997 multimodal math problems pair worked solutions with visual keypoints that an explanation must reference. | Evaluation found models often missed or misused the visual evidence needed to explain a solution like a human mentor. |
| [CSTutorBench](https://arxiv.org/abs/2607.05571) | Middle-school block-programming tutoring cases probe answer leakage and whether tutors use learner debugging history. | Tested models frequently disclosed answers or failed to adapt their guidance to earlier debugging evidence. |
| [StudentBench](https://arxiv.org/abs/2609.28470) · [study data/code](https://github.com/Handshake-AI-Research/studentbench) · [dataset](https://huggingface.co/datasets/handshake-ai-research/studentbench) | Public, de-identified GRE study data include pre/post assessments, 176K tutoring messages, practice attempts, tutor reviews, surveys, and cost data from 2,383 participants. | The preprint reports AI tutoring gains statistically equivalent to expert human tutoring and one AI condition at equivalent gains for 918× lower cost, in this GRE study. |
| [Assistance Dilemma / near-transfer set](https://arxiv.org/abs/2610.06446) | Fresh BigMath-derived source/target problem pairs test whether a tutor lets a student attempt a related problem before assisting; uses a frozen simulated learner. | The authors report that a masked-context gating policy reduced unnecessary handovers and improved simulated near-transfer performance, including out-of-distribution cases. |
| [L2-Bench](https://arxiv.org/abs/2607.08842) · [dataset](https://huggingface.co/datasets/OUP/l2-bench) | 1,000+ expert-reviewed second-language education tasks with a 12-competency/31-subcompetency taxonomy and rubric-scored reference responses. | Nine evaluated models showed a top overall score of 85.5%, with performance dropping on harder language-teaching tasks. |
| [Teachometry benchmark](https://teachometry.com/data/) · [method](https://teachometry.com/methodology/) | Developer-preview open set of 48 authored cases across five subjects, five evaluation dimensions, 27 tutoring capabilities, and English/Simplified Chinese. | The public site reports no calibrated model trials yet, so it currently offers inspectable cases and rubrics rather than comparative evidence. |
| [EduAgentBench](https://huggingface.co/datasets/eduagentbench/eduagentbench) | 150-task research release covering pedagogical judgment, situated multi-turn tutoring, and simulated learning-management-system workflows; its card describes it as a 2026 NeurIPS Evaluations and Datasets Track submission. | The public release provides a task suite but no peer-reviewed comparative model results, so treat scores as exploratory. |

### Broader educational capability benchmarks

These are useful for checking knowledge, age-fit, instructional goals, or learning-material generation, but they are not all tests of interactive tutor behavior.

| Resource and primary source | What it contains / access | Paper or experiment result (one sentence max) |
|---|---|---|
| [Fab AI Pedagogy Benchmark / CDPK and SEND](https://arxiv.org/abs/2506.18710) · [official repo](https://github.com/AI-for-Education/pedagogy-benchmark) · [leaderboard](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/benchmarks) | 1,143 Chilean teacher-exam items translated into English: 920 cross-domain pedagogy questions and a 223-item SEND subset. | The paper evaluated 97 models and reported 28–89% accuracy, measuring pedagogical knowledge rather than whether a tutor teaches a learner effectively. |
| [Fab AI Visual Maths](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/resources/benchmarks/about-the-visual-maths-benchmark) · [parent MATH-Vision set](https://arxiv.org/abs/2402.14804) | Fab AI’s 237 early-grade visual-math items are selected from MATH-Vision, a larger image-based math benchmark; the subset focuses on pictorial arithmetic and diagram interpretation. | Fab AI found that model failures were driven more by visual interpretation than by the arithmetic itself. |
| [MATH-Vision](https://arxiv.org/abs/2402.14804) | Image-based benchmark of 3,040 mathematical questions; it is the parent pool from which Fab AI selected its early-grade Visual Maths subset. | The benchmark tests visual mathematical reasoning that text-only math sets miss, while Fab AI’s subset shows that visual interpretation is a major failure source. |
| [Fab AI Visual Reasoning Benchmark](https://arxiv.org/abs/2602.12196) · [official page](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/resources/benchmarks/about-the-visual-reasoning-benchmark) | 701 nonverbal pattern, matching, and spatial-reasoning questions from primary exams in Zambia and India. | The paper reports a spatial-reasoning ceiling: models handle static visual tasks better than folding, reflection, and rotation. |
| [Fab AI Multilingual Pedagogy](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/resources/research-paper/talking-teachers-language) · [language leaderboard](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/benchmarks) | Pedagogy-test translations and model evaluations cover Arabic, Kiswahili, Hausa, Pashto, Yoruba, Dari, Luganda, and Nyankore. | The study and leaderboard show that English pedagogical performance does not guarantee equivalent performance in the language teachers use in class. |
| [Fab AI Linguistic Knowledge](https://huggingface.co/datasets/AI-for-Education/English-Linguistic-Knowledge-Benchmark) | Tests knowledge of orthography, phonology, morphology, syntax, semantics, and reading-related language structure; public English release is a 20-item sample of a 137-item benchmark. | Fab AI’s release provides the benchmark and leaderboard as a diagnostic for foundational literacy; the sample alone is too small to treat as a complete evaluation. |
| [Fab AI Decodability Assessment](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/benchmarks/about) | Tests whether a model can identify words a beginning reader can decode given the letter-sound sequence already taught; Fab reports English and Kiswahili versions. | The public launch establishes a new foundational-literacy evaluation, but I found no peer-reviewed comparative result by the cutoff. |
| [Dr.Academy](https://aclanthology.org/2024.acl-long.173/) | Benchmark of educational question generation aligned with subject context and cognitive/educational taxonomies. | GPT-4 performed strongly on general and single-discipline tasks, while Claude 2 did better on interdisciplinary question generation. |
| [EducationQ](https://arxiv.org/abs/2504.14928) · [data/code](https://github.com/SunriserFuture/EducationQ) | Multi-agent teaching simulation evaluates 1,498 tasks across 13 disciplines, with pre/post learner questions and qualitative tutoring analysis. | Teaching effectiveness did not scale linearly with model size or reasoning ability, and some smaller open models outperformed larger commercial models. |
| [EduBench](https://arxiv.org/abs/2505.16160) · [data/code](https://github.com/ybai-nlp/EduBench) | Synthetic benchmark with 4,000+ educational contexts across nine scenarios and 12 evaluation metrics with human annotation. | The authors trained a smaller model that approached DeepSeek V3 and Qwen Max on the benchmark test set. |
| [OpenLearnLM Benchmark](https://arxiv.org/abs/2601.13882) | 124K+ items test knowledge, scenario-based skills, and attitude/alignment consistency across educational roles and subjects. | Seven frontier models had different strengths and weaknesses, with no single model leading on every dimension. |
| [K-12EduBench](https://ojs.aaai.org/index.php/AAAI/article/view/40744) | 3,259 questions across nine subjects, plus IRT models, multi-step solution scoring, and cognitive-level alignment measures. | Education-optimized models outperformed general models across the three measured abilities, yet all still fell short of real instructional needs. |
| [EduAdapt](https://aclanthology.org/2025.emnlp-main.1736/) · [data/code](https://github.com/NaumanNaeem/EduAdapt) | Nearly 48K grade-labeled science QA pairs across nine subjects and grades 1–12 test grade-appropriate explanations. | Larger models generally did better but still struggled to produce suitable responses for early grades 1–5. |
| [LessonBench-V1](https://arxiv.org/abs/2607.13041) | Lesson-generation benchmark records 3,620 learning objectives with pedagogical metadata for evaluating lesson-generation agents. | The paper introduces the task set and evaluation setup; it is a lesson-planning benchmark rather than a test of tutoring dialogue or student learning. |
| [Professional Competency Benchmark for LLMs](https://arxiv.org/abs/2411.00045) | Psychometrics-based teacher/professional competency benchmark for evaluating educational knowledge and applied judgment. | The authors report that models show promise but remain unreliable on tasks requiring deeper cognitive and pedagogical engagement. |
| [EduEVAL-DB](https://arxiv.org/abs/2602.15531) | Educational explanations paired with human/simulated-teacher references and labels for correctness, depth, relevance, age-fit, and bias. | The paper presents preliminary risk-evaluation experiments, so its current results are best used to design review rubrics rather than rank tutors. |
| [Education Dialogue / MTPO dataset](https://arxiv.org/abs/2405.14655) · [official release](https://github.com/google-research-datasets/Education-Dialogue-Dataset) | About 40K synthetic training dialogues and 7,234 test dialogues generated with Gemini Ultra for multi-turn preference optimization. | MTPO beat single-turn RLHF baselines on the education dialogue task, while the paper’s larger preference-only comparison also matched explicit reward modeling in a separate domain. |

### Tutoring dialogue and training corpora

These resources are especially useful for designing club cases, tutor-move labels, and practice data; their synthetic-student or simulated-dialogue components should be validated against real learners before being used as a reward signal.

| Resource and primary source | What it contains / access | Paper or experiment result (one sentence max) |
|---|---|---|
| [MathDial](https://arxiv.org/abs/2305.14536) · [dataset/code](https://github.com/eth-nlped/mathdial) | About 3K multi-turn math dialogues pairing human tutors with LLM students prompted to make realistic errors; turns have pedagogical annotations. | MathDial enabled fine-tuning and evaluation of the help-versus-answer-leakage trade-off, but its simulated students do not establish real learner outcomes. |
| [Bridge](https://arxiv.org/abs/2310.10648) · [dataset/code](https://github.com/rose-e-wang/bridge) | About 700 elementary math tutoring dialogues between real students and human tutors, with expert-coded strategies, intentions, and error types. | In the Bridge paper, GPT-4 responses conditioned on expert-selected strategies were preferred 76% more than responses without those decisions, while random strategy choices reduced preference sharply. |
| [CIMA](https://aclanthology.org/2020.bea-1.5/) · [dataset/code](https://github.com/kstats/CIMA) | 741 crowd-role-played Italian-learning exercises and 5,850 tutor responses, including three plausible reference responses and dialogue-action labels per context. | Student role-players completed exercises in fewer turns, and a neural response generator achieved BERT F1 0.53 versus 0.45 for the rule-based baseline while retaining similar BLEU. |
| [CoMTA](https://github.com/Khan/tutoring-accuracy-dataset) · [Miller & DiCerbo paper](https://doi.org/10.35542/osf.io/5zwv3) | 188 anonymized real-use Khanmigo/student math dialogues, elementary math through calculus, with answer-grounded accuracy evaluation; license restricts use to internal, non-commercial model evaluation and prohibits training or redistribution. | The paper compares model mathematical accuracy in tutoring contexts; later dialogue knowledge-tracing work used CoMTA with MathDial and reported better student-response prediction than its tested KT baselines. |
| [Eedi Question-Anchored Tutoring Dialogues 2K](https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k) | 1,971 real math tutoring sessions anchored to questions, with tutor/student turns and tutor talk-move labels; released under CC BY-NC 4.0. | The release enables analysis of authentic student uptake and tutor moves, but the dataset card does not report a causal learning-outcome experiment. |
| [SocraTeach / SocraticLM data](https://papers.nips.cc/paper_files/paper/2024/hash/9bae399d1f34b8650351c1bd3692aeae-Abstract-Conference.html) · [code/data](https://github.com/Ljyustc/SocraticLM) | 35K generated multi-round Socratic math dialogues with six simulated learner cognitive states. | Fine-tuning improved Socratic teaching scores across five rubric dimensions, but the paper does not demonstrate real-student learning gains. |
| [SocraticMATH](https://arxiv.org/abs/2407.17349) · [dataset](https://github.com/ECNU-ICALK/SocraticMath) | 6,846 Chinese primary-math Socratic dialogues covering 513 concepts and multiple teaching moves. | The authors report that SocraticLLM improved teaching-quality and reasoning-reliability measures over comparison models. |
| [SocratDataset / KELE](https://aclanthology.org/2025.findings-emnlp.888/) | More than 42K turns organized around 34 structured teaching strategies in the KELE multi-agent framework. | The authors report that SocratTeachLLM outperformed GPT-4o on their Socratic-teaching evaluation. |
| [CLASS datasets](https://aclanthology.org/2023.findings-emnlp.130/) | Scaffolding and dialogue corpora for SPOCK, an introductory-biology ITS, built with the learning-science-based CLASS framework. | Biology subject experts gave favorable preliminary judgments of SPOCK’s stepwise guidance and encouraging responses; this was not a student outcome trial. |
| [EduDial](https://arxiv.org/abs/2510.12899) · [data/code](https://github.com/ECNU-RAIL/EduDial-EMNLP2026) | 34,250 generated teacher–student math dialogue sessions cover 345 K–12 knowledge points, ten questioning strategies, and 13,700 SFT dialogues plus 20,550 preference pairs. | EduDial-LLM 32B outperformed 17 baseline models across the paper’s 11 teaching and content-quality metrics. |
| [ConvoLearn](https://arxiv.org/abs/2601.08950) · [dataset](https://huggingface.co/datasets/masharma/convolearn) | Semi-synthetic, teacher-authored middle-school Earth Science dialogues labeled on six dialogic teaching dimensions; the latest release reports 2,134 dialogues. | Teacher ratings favored fine-tuned Mistral-7B over its base model and Claude Sonnet 4.5 on instructional quality, without testing student learning. |
| [GeoDial](https://arxiv.org/abs/2606.12419) | More than 1.3K human geometry tutoring dialogues annotated with diagram highlights and step-level teaching actions. | Fine-tuning improved tutoring language, but models still struggled to highlight the right diagram regions. |
| [GMADE / Math Assistant dialogue study](https://aclanthology.org/2025.findings-emnlp.605/) | German-language math-assistant conversations used to study prompt-moderated LLM tutoring at tertiary level. | The examined prompt-moderation strategy did not reliably solve the tutoring problems observed in the conversations. |
| [MathDial/CoMTA knowledge annotations](https://github.com/umass-ml4ed/dialogue-kt) | Adds per-turn knowledge-component and correctness labels to two dialogue datasets for learner-state modeling. | The associated study reported that LLMKT predicted student response correctness better than the knowledge-tracing baselines it tested. |
| [EduAlign HPC examples](https://arxiv.org/abs/2507.20335v1) | The v1 paper builds an 8K educational-QA set with expert/model labels for helpfulness, personalization, and creativity reward modeling. | GRPO training improved the three reward-model scores in v1, while the revised v2 changes the optimization method and training-data recipe. |
| [TeachLM authentic tutoring corpus](https://arxiv.org/abs/2510.05087) | Private, anonymized Polygence archive of 100,000 hours of one-to-one longitudinal tutoring; the raw student data are not public. | Fine-tuning on this corpus doubled student talk time, increased dialogue turns by 50%, and improved questioning and personalization in the authors’ evaluations. |
| [StudyChat programming-tutor conversations](https://arxiv.org/abs/2607.09919) | Public naturalistic student–ChatGPT programming conversations with annotations for help-seeking, student state, tutor response style, and next-turn outcome. | LLM-assisted labels reached 82% human agreement, and the authors found that tutoring effectiveness depends on the learner’s context and next-turn uptake. |
| [Socratic-bench](https://github.com/GiovanniGatti/socratic-bench) | Community research release with expert-labeled judge-calibration data, generated teacher–student evaluation dialogues, and successful Socratic traces for fine-tuning. | On its own benchmark, Qwen3-4B fine-tuned on 754 successful traces scored 22.8% versus 17.2% for the prompted base model; this is a repository-reported result, not a peer-reviewed trial. |

### Learner-state and adaptive-tutoring datasets

These are useful if the club needs to infer what a learner knows or choose a next exercise; most are interaction logs, not datasets of high-quality tutoring language.

| Resource and primary source | What it contains / access | Paper or experiment result (one sentence max) |
|---|---|---|
| [FoundationalASSIST](https://arxiv.org/abs/2602.00070) | 1.7M ASSISTments interactions from 5,000 students, with full item text, actual responses and wrong-answer choices, and Common Core alignment. | Frontier LLMs barely beat a trivial knowledge-tracing baseline and fell below chance on identifying discriminative assessment items. |
| [KT-PSP-25](https://arxiv.org/abs/2512.00311) · [ACL paper](https://aclanthology.org/2026.findings-acl.961/) · [data/code](https://github.com/jungyangpark/KT-PSP-25) | Dataset of 22,289 math interactions from 1,343 students, including their written problem-solving processes, designed to represent proficiency beyond right/wrong outcomes. | The StatusKT teacher–student–teacher pipeline improved knowledge-tracing prediction on KT-PSP-25 and produced interpretable proficiency indicators. |
| [ASSISTments public data](https://sites.google.com/view/assistmentsdatamining/dataset) | Longitudinal math practice data, including commonly used 2009, 2012, and 2017 releases, with responses, skills, attempts, hints, and problem metadata. | Knowledge-tracing papers use ASSISTments to predict future student responses, but those prediction scores measure state estimation rather than tutor quality. |
| [KDD Cup 2010 educational data](https://pslcdatashop.web.cmu.edu/KDDCup/) | Public student-step and skill-tagged interaction logs from Algebra and Bridge to Algebra tutoring systems. | Deep Knowledge Tracing and later KT models use KDD Cup data to compare future-response prediction, not dialog pedagogy or causal learning gains. |
| [EdNet](https://arxiv.org/abs/1912.03072) · [data](https://github.com/riiid/ednet) | Large Korean online-learning interaction corpus with four nested sets and over 131M interactions in its largest release. | The release evaluates knowledge-tracing baselines on next-response prediction, offering scale for student-state modeling but no tutor-quality result. |
| [MOOCRadar](https://arxiv.org/abs/2304.02205) · [code/data](https://github.com/THU-KEG/MOOC-Radar) | MOOC knowledge repository with 2,513 questions, 5,600 concepts, and more than 12 million learner-behavior records; LongTutor uses processed sequences from it. | The original study reports the resource supports student modeling, while LongTutor later found models used historical evidence less effectively for mastery diagnosis and teaching-action choice. |
| [XES3G5M](https://arxiv.org/abs/2205.12333) | Large-scale K–12 math exercise and student-response dataset with knowledge-component structure; LongTutor uses its processed traces. | LongTutor found that access to extensive XES3G5M-derived histories did not remove models’ diagnosis and adaptive-action weaknesses. |
| [Junyi Academy Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy) | Public math-platform logs used in knowledge-tracing research, with student problem attempts and knowledge-component metadata. | KT studies use Junyi to test student-response prediction; its accuracy metrics do not directly establish that an adaptive tutor improves learning. |
| [EvalConvoLearn](https://arxiv.org/abs/2608.07497) · [code](https://github.com/RenaissancePhilanthropy/EvalConvoLearn) | Open learner-simulator evaluation framework checks both skill-conditioned mastery change and conversational behavior against real tutoring data. | The demonstration on tutoring data shows that two LLM learner simulators differ across learning and conversational measures, underscoring that fluent dialogue alone is not simulator validity. |

### Other useful domain-specific and privacy resources

| Resource and primary source | What it contains / access | Paper or experiment result (one sentence max) |
|---|---|---|
| [MathEd-PII](https://arxiv.org/abs/2602.16571) · [EDM proceedings](https://educationaldatamining.org/edm2026/proceedings/2026.EDM.full-papers.183/index.html) | Math-chat examples annotated for student personally identifiable information and redaction quality. | Math-aware prompting improved PII detection over the generic baseline, making this a practical check before saving club dialogue histories. |
| [CIMA dialogue-act evaluation](https://arxiv.org/abs/2509.09125) | Reuses the CIMA tutoring corpus to test automatic classification of tutor dialogue acts. | GPT-4 reached 80% accuracy and weighted F1 0.81, with substantial agreement against human labels. |

**Practical shortlist for the club:** Start with MathTutorBench, GuideEval, TutorMoments, and MRBench to write behavior checks; use Eedi QATD2k, Bridge, or CIMA for real conversational patterns; use FoundationalASSIST only if you are building learner-state estimates; and treat StudentBench as an example of outcome measurement, not as evidence that the same intervention will work for your club. If you simulate students, check simulator fidelity with SimulatorArena or EvalConvoLearn before using simulated outcomes as rewards. Keep benchmark scores, learner self-reports, and unaided performance on fresh problems as separate measures.

## Suggested path for the student club

1. **Define the learning context:** Pick one club subject, learning goals, and a small prerequisite skill map. Decide when the tutor should hint, explain, or show a solution; use Knowledge Tracing and LongTutor as references for storing evidence, not personality labels.
2. **Prototype a teaching sequence:** Start with LearnLM-style instructions and a compact intent vocabulary from MathDial and Intent Matters. Try an explicit dialogue plan (StratL), self-explanation prompts and faded examples (Atkinson et al.), then retry and mastery checks (Oreopoulos et al.).
3. **Evaluate behavior and learning separately:** Build club-authored cases for accuracy, misconception diagnosis, hint quality, context-sensitive action changes, and answer leakage. Use MathTutorBench, MRBench, KMP-Bench, TutorBench, SHAPE, and the adversarial leakage benchmark for rubric ideas; include fresh unaided problems based on the harm and transfer evidence from Bastani et al. and classroom trials.
4. **Pilot in the real club workflow:** Test whether students ask for help, act on it, and return for another attempt. Tutor CoPilot and LearnLM/Eedi support human-reviewed designs; Khanmigo and Kobler et al. show why actual uptake and use patterns matter.
5. **Add adaptation before model training:** First adapt exercise sequence or hint timing to observed attempts, as in Chung et al. and Alam et al. If prompting and workflow changes are insufficient, consider supervised fine-tuning or preference training on mentor-reviewed cases. Use simulator-driven RL only after validating simulator fidelity; the PedagogicalRL papers and reward-model work are later-stage options.

**Evaluation caution:** A simulated student’s next-turn correctness is only a proxy for learning. Whenever feasible, use a fresh problem or explanation task after the dialogue, without the tutor answer in view, and ask the learner whether the interaction helped. Treat benchmark or model-judge scores as diagnostic signals rather than proof of educational impact.

## Primary-source index

- [PedagogicalRL, arXiv:2505.15607](https://arxiv.org/abs/2505.15607)
- [PedagogicalRL-Thinking, arXiv:2601.14560](https://arxiv.org/abs/2601.14560)
- [EduAlign current v2, arXiv:2507.20335](https://arxiv.org/abs/2507.20335) and [original v1](https://arxiv.org/abs/2507.20335v1)
- [Scarlatos et al., arXiv:2503.06424](https://arxiv.org/abs/2503.06424)
- [SocraticLM, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/9bae399d1f34b8650351c1bd3692aeae-Abstract-Conference.html)
- [MathTutorBench, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.11/)
- [MathDial, arXiv:2305.14536](https://arxiv.org/abs/2305.14536)
- [Intent Matters, BEA 2025](https://aclanthology.org/2025.bea-1.63/)
- [LearnLM, arXiv:2412.16429](https://arxiv.org/abs/2412.16429)
- [Bloom, “The 2 Sigma Problem” (1984)](https://doi.org/10.3102/0013189X013006004)
- [Aleven & Koedinger, Cognitive Tutor self-explanation study (2002)](https://doi.org/10.1207/s15516709cog2602_1)
- [VanLehn, tutoring-effectiveness review (2011)](https://doi.org/10.1080/00461520.2011.611369)
- [Ma et al., ITS learning-outcomes meta-analysis (2014)](https://doi.org/10.1037/a0037123)
- [Sonkar et al., Pedagogical Alignment (EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.797/)
- [Maurya et al., MRBench (NAACL 2025)](https://aclanthology.org/2025.naacl-long.57/)
- [Cohn et al., Adaptive Scaffolding (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/37154)
- [Kestin et al., AI tutoring RCT (Scientific Reports 2025)](https://www.nature.com/articles/s41598-025-97652-6)
- [Oreopoulos & Low, Khanmigo school experiment (NBER 2026)](https://www.nber.org/papers/w35620)
- [Wang et al., Tutor CoPilot (arXiv:2410.03017)](https://arxiv.org/abs/2410.03017)
- [Shi et al., KMP-Bench (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/40578)
- [Bastani et al., generative AI and math learning outcomes (PNAS 2025)](https://doi.org/10.1073/pnas.2422633122)
- [Kraft et al., tutoring effects at scale (Review of Educational Research 2026)](https://journals.sagepub.com/doi/10.3102/00346543261446660)
- [Corbett & Anderson, Knowledge Tracing](https://link.springer.com/article/10.1007/BF01099821)
- [Alam et al., adaptive subgoal hints (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26887)
- [Atkinson et al., faded worked examples and self-explanation](https://doi.org/10.1037/0022-0663.95.4.774)
- [Puech et al., StratL pedagogical steering (ACL 2025)](https://aclanthology.org/2025.findings-acl.1348/)
- [Chung et al., adaptive AI tutoring via RL (arXiv:2608.16907)](https://arxiv.org/abs/2608.16907)
- [Worden et al., large-scale AI-generated feedback RCT (Learning @ Scale 2026)](https://doi.org/10.1145/3774398.3811594)
- [Oreopoulos et al., AI tutoring and mastery practice (NBER 35621)](https://www.nber.org/papers/w35621)
- [LearnLM/Eedi classroom RCT (arXiv:2512.23633)](https://arxiv.org/abs/2512.23633)
- [De Simone et al., Nigeria AI tutoring study (World Bank 2025)](https://documents1.worldbank.org/curated/en/099548105192529324/pdf/IDU-c09f40d8-9ff8-42dc-b315-591157499be7.pdf)
- [Borchers & Shou, LLM tutoring adaptivity benchmark](https://arxiv.org/abs/2504.05570)
- [Li et al., LongTutor (ACL 2026)](https://aclanthology.org/2026.acl-long.1371/)
- [Scarlatos et al., simulated student fidelity (ACL 2026)](https://aclanthology.org/2026.acl-long.1960/)
- [Srinivasa et al., TutorBench](https://arxiv.org/abs/2510.02663)
- [Yang et al., MMTutorBench (ACL 2026)](https://aclanthology.org/2026.acl-long.1068/)
- [Zhao et al., SHAPE educational safety benchmark (ACL 2026)](https://aclanthology.org/2026.acl-long.529/)
- [Zhao et al., adversarial answer-leakage benchmark (ACL 2026)](https://aclanthology.org/2026.acl-long.1412/)
- [Petukhova & Kochmar, reward modeling for math mistake remediation (LREC 2026)](https://aclanthology.org/2026.lrec-1.232/)
- [Kobler et al., real student use of LLMs (ACL 2026)](https://aclanthology.org/2026.acl-long.875/)
