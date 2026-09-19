# Document Intelligence Assistant — Evaluation Pipeline

> **A Complete, Friendly Guide for Absolute Beginners**  
> *(Even if you have never heard of AI, computers, or robots before!)*

---

## Table of Contents

1. [Part 1 — Start from Zero: What Does "Evaluation" Mean?](#part-1--start-from-zero-what-does-evaluation-mean)
2. [Part 2 — What Are We Evaluating?](#part-2--what-are-we-evaluating)
3. [Part 3 — What Is RAG? (Retrieval-Augmented Generation)](#part-3--what-is-rag)
4. [Part 4 — What Is an LLM? (Large Language Model)](#part-4--what-is-an-llm)
5. [Part 5 — What Is an Embedding? (The Meaning Map)](#part-5--what-is-an-embedding)
6. [Part 6 — What Is ChromaDB? (The Ephemeral Memory Box)](#part-6--what-is-chromadb)
7. [Part 7 — What Is Retrieval? (Finding the Right Pages)](#part-7--what-is-retrieval)
8. [Part 8 — What Is BM25? (The Word-Matcher) and Hybrid Search](#part-8--what-is-bm25-and-hybrid-search)
9. [Part 9 — What Is LangGraph? (The Smart Teacher's Flowchart)](#part-9--what-is-langgraph)
10. [Part 10 — What Is Query Rewriting?](#part-10--what-is-query-rewriting)
11. [Part 11 — What Is a Hallucination? (Making Things Up)](#part-11--what-is-a-hallucination)
12. [Part 12 — What Is Grounding? (Pointing to the Proof)](#part-12--what-is-grounding)
13. [Part 13 — What Is an Evaluation Dataset?](#part-13--what-is-an-evaluation-dataset)
14. [Part 14 — What Is CUAD? (Contract Understanding Atticus Dataset)](#part-14--what-is-cuad)
15. [Part 15 — Retrieval Evaluation (Did We Find the Proof?)](#part-15--retrieval-evaluation)
16. [Part 16 — Why Retrieval Evaluation Is Crucial](#part-16--why-retrieval-evaluation-is-crucial)
17. [Part 17 — What Is Ragas? (The Automated Grader)](#part-17--what-is-ragas)
18. [Part 18 — Faithfulness (Did the Robot Tell the Truth?)](#part-18--faithfulness)
19. [Part 19 — Context Precision (Are Good Clues at the Top?)](#part-19--context-precision)
20. [Part 20 — Context Recall (Did We Find All Needed Clues?)](#part-20--context-recall)
21. [Part 21 — Answer Relevancy (Did the Robot Answer the Real Question?)](#part-21--answer-relevancy)
22. [Part 22 — Putting the Ragas Metrics Together](#part-22--putting-the-ragas-metrics-together)
23. [Part 23 — What 90% Really Means (And What It Does NOT Mean)](#part-23--what-90-really-means)
24. [Part 24 — Complete System & Evaluation Pipeline (Master Diagrams)](#part-24--complete-system--evaluation-pipeline)
25. [Part 25 — Two Types of Testing: Runtime Protection vs Offline Evaluation](#part-25--two-types-of-testing)
26. [Part 26 — Out-of-Domain Evaluation (Saying "I Don't Know")](#part-26--out-of-domain-evaluation)
27. [Part 27 — Hallucination Evaluation (Testing Blank Spaces)](#part-27--hallucination-evaluation)
28. [Part 28 — Walkthrough of a Good Evaluation Example](#part-28--walkthrough-of-a-good-evaluation-example)
29. [Part 29 — Walkthrough of a Bad / Failure Example](#part-29--walkthrough-of-a-bad--failure-example)
30. [Part 30 — How to Read the Actual Evaluation Report](#part-30--how-to-read-the-actual-evaluation-report)
31. [Part 31 — Troubleshooting Guide: What Each Failure Means](#part-31--troubleshooting-guide-what-each-failure-means)
32. [Part 32 — Why Not Just Ask a Human?](#part-32--why-not-just-ask-a-human)
33. [Part 33 — Real-World Limitations](#part-33--real-world-limitations)
34. [Part 34 — One-Page Cheat Sheet](#part-34--one-page-cheat-sheet)
35. [Part 35 — Explain It to a Child in 60 Seconds](#part-35--explain-it-to-a-child-in-60-seconds)
36. [Part 36 — Current Implementation vs Original Plan](#part-36--current-implementation-vs-original-plan)

---

## Part 1 — Start from Zero: What Does "Evaluation" Mean?

### The Story of the Book-Reading Robot 🤖📖

Imagine you build a little friendly toy robot.  
You give the robot a big, heavy storybook.

You do not simply look at the robot and say:  
*"Wow, it looks shiny! It must be working!"*

Instead, you ask the robot questions:
- **Did it open the right page?** *(Or did it look at pictures of dogs when you asked about cats?)*
- **Did it find the right words?** *(Did its eyes land on the important sentence?)*
- **Did it tell you the truth?** *(Or did it make up a fairy tale that is not in the book?)*
- **Did it understand what you asked?** *(When you asked for the time, did it tell you the weather?)*
- **Did it say "That's not in the book!" when asked about something else?** *(If you ask about chocolate cake, and the book is about spaceships, does it know to say "I don't know"?)*

```
       WITHOUT EVALUATION                      WITH EVALUATION
   ┌──────────────────────────┐           ┌──────────────────────────┐
   │ Robot says: "Blue cows!" │           │ Does the book say cows   │
   │                          │   ───►    │ are blue?                │
   │ Human: "Sounds cool!"    │           │                          │
   │ (Danger: Robot lied!)    │           │ ✗ NO! Cow is white/brown.│
   └──────────────────────────┘           │ Robot failed truth check!│
                                          └──────────────────────────┘
```

### In Simple Words:
> **Evaluation** means **giving our robot a test** to measure whether it is doing its job accurately, truthfully, and reliably.

---

## Part 2 — What Are We Evaluating?

Our project is called the **Document Intelligence Assistant**.  
Think of it as a super-attentive student sitting at a desk with a single PDF contract document.

```
       WHAT THE STUDENT DOES STEP-BY-STEP:

       1. Reads the PDF page by page (Extracts text and tables)
                            ↓
       2. Cuts the pages into smaller index cards ("Chunks")
                            ↓
       3. Gives every card a number fingerprint ("Embeddings")
                            ↓
       4. Puts all cards in a temporary box ("Ephemeral ChromaDB")
                            ↓
       5. User asks: "What are the payment terms?"
                            ↓
       6. Student searches the box and pulls out the 5 best cards
                            ↓
       7. Student checks: "Do these cards actually answer the question?"
                            ↓
       8. If the cards are missing clues, student rewrites the search
          query in clearer legal words and searches again (Max 2 rewrites)
                            ↓
       9. Writes a clear, simple answer in plain English
                            ↓
      10. Proofreads the answer: "Is every single sentence backed up
          by the words on the cards?" (Grounding check)
                            ↓
      11. Shows the verified answer with exact page citations!
```

```
+-------------------------------------------------------------------------+
|                        FULL PROCESSING FLOWCHART                        |
+-------------------------------------------------------------------------+

      [ PDF Upload ] 
            │
            ▼
      [ Read PDF ] ─── (pdfplumber + pypdf)
            │
            ▼
      [ Extract Chunks ] ─── (Parent: ~1000 chars, Child: ~800 chars)
            │
            ▼
      [ Create Embeddings ] ─── (sentence-transformers/all-MiniLM-L6-v2)
            │
            ▼
      [ In-Memory ChromaDB + BM25 ] ─── (RAM Only, Ephemeral)
            │
            ▼
      [ User Question ]
            │
            ▼
      [ Hybrid Retrieval ] ─── (Dense 50% + BM25 50% -> Top 5)
            │
            ▼
      [ LangGraph Grade Node ]
            ├── Context Good? ──► [ Generate Answer ]
            └── Context Bad?  ──► [ Rewrite Query ] (Max 2 times)
                                        │
                                        └──► [ Retrieve Again ]
                                                   │
                                                   └──► [ Generate Answer ]
                                                              │
                                                              ▼
                                                   [ Grounding Check ]
                                                              │
                                                              ▼
                                                   [ Final Answer + Citations ]
```

---

## Part 3 — What Is RAG?

**Technical Term:** RAG  
**Full Name:** Retrieval-Augmented Generation  
**Simple Meaning:** *"Look up the facts in the book first, THEN write the answer!"*

Let's break down the three words:

| Word | What it Means in Plain English |
|---|---|
| **Retrieval** | Searching the document to find the exact pages that have the clues. |
| **Augmented** | Handing those exact pages to the writer so it has the evidence in front of its eyes. |
| **Generation** | Letting the writer compose a clear, easy-to-read answer. |

### Why Does This Matter? (The Closed-Book vs Open-Book Exam)

- **Without RAG (Closed-Book Exam):**  
  You ask the AI: *"What is the termination period of Company ABC's agreement?"*  
  The AI has to guess from its memory. Since it has never seen Company ABC's private contract, it might make up a random number like *"60 days"*. That is dangerous!

- **With RAG (Open-Book Exam):**  
  1. We search Company ABC's contract.  
  2. We find Clause 14 on Page 8: *"Either party may terminate upon thirty (30) days prior notice."*  
  3. We place that paragraph right in front of the AI.  
  4. The AI writes: *"According to Clause 14 on Page 8, the termination notice period is 30 days."*

---

## Part 4 — What Is an LLM?

**Technical Term:** LLM  
**Full Name:** Large Language Model  
**Simple Meaning:** A giant computer brain that is very good at reading, explaining, and writing human language.

### Analogy:
Imagine an author who has read millions of English grammar books, dictionaries, encyclopedias, and essays. This author knows how to speak smoothly, explain complex sentences, and summarize ideas.

### The Catch:
**The LLM does NOT know what is written in your private uploaded PDF!**  
It has never seen your private contract before.

```
   NORMAL LLM (No RAG):
   User Question ──────► LLM ──────► Guess / Common Knowledge (May be wrong!)

   OUR SYSTEM (With RAG):
   User Question ──────► Search Document ──────► Exact Evidence ──────► LLM ──────► Grounded Truth
```

In our project:
- **Primary LLM:** Groq API using high-speed `openai/gpt-oss-120b` (or OpenAI `gpt-4o-mini`).
- **Offline Fallback:** Deterministic rule-based extractor so tests run even without internet or API keys!

---

## Part 5 — What Is an Embedding?

**Technical Term:** Embedding  
**Simple Meaning:** A list of numbers that acts like a GPS location for meaning.

### The Word Map Analogy 🗺️
Imagine a giant playground map.
- "Apple" sits next to "Banana" and "Orange" (Fruit corner).
- "Dog" sits next to "Puppy" and "Wolf" (Animal corner).
- "Airplane" sits far away next to "Helicopter" (Vehicle corner).

Computers do not understand letters like humans do. But computers LOVE numbers!  
An **embedding model** turns sentences into a string of numbers (vectors):

```
"Termination notice period"   ──►  [ 0.12, -0.45,  0.78,  0.03, ... ]
"How to cancel the agreement" ──►  [ 0.11, -0.42,  0.76,  0.05, ... ]
"Recipe for chocolate cake"   ──►  [-0.89,  0.92, -0.04, -0.61, ... ]
```

Notice how *"Termination notice period"* and *"How to cancel the agreement"* have almost identical numbers!  
Even though they use completely different words, the computer knows they mean the same thing!

In our project:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Runs directly on your CPU (no cloud API needed, completely free and private).

---

## Part 6 — What Is ChromaDB?

**Technical Term:** ChromaDB  
**Simple Meaning:** A magical filing cabinet that organizes document pieces by their meaning.

When a user asks:  
*"What happens if I want to end the contract early?"*

ChromaDB takes that question, checks the number-coordinates, and instantly pulls out the document pieces sitting in the same corner of the meaning map.

### What Does "Ephemeral" Mean?
Our implementation uses `chromadb.EphemeralClient()`.

> **Ephemeral = Temporary like morning dew! 💧**

- **In-Memory Only:** The filing cabinet lives only in computer RAM while your document is open.
- **Zero Disk Traces:** It creates **no** SQLite database files, **no** storage folders, and **no** permanent cache.
- **Automatic Clean-Up:** As soon as you upload a new document or shut down the server, the old index vanishes completely. Zero risk of leaking confidential contract data!

---

## Part 7 — What Is Retrieval?

**Technical Term:** Retrieval  
**Simple Meaning:** Searching through all the document pieces to find the most useful clues.

### What is "Top-K"?
When the user asks a question, the retriever could find 50 different mentions of words. But we don't want to overwhelm the AI with 50 pages!

If **K = 5**:
We tell the retriever:  
*"Give me the **5 best, most relevant pieces** of evidence."*

```
                 QUESTION: "What is the warranty period?"
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │                  DOCUMENT RETRIEVER                    │
       └────────────────────────────────────────────────────────┘
            │               │               │               │
            ▼               ▼               ▼               ▼
         Rank 1          Rank 2          Rank 3          Rank 4
        [Page 12]       [Page 13]       [Page 4]        [Page 22]
       "Warranty is    "Exceptions     "Payment for    "General
       12 months..."   to warranty"    spare parts"    definitions"
            ▲               ▲
            └───────┬───────┘
     TOP CLUES GIVEN TO THE LLM!
```

---

## Part 8 — What Is BM25 and Hybrid Search?

In our Document Intelligence Assistant, we do not rely on just one search method. We combine **two different search superpowers**:

```
 ┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
 │       SUPERPOWER 1: DENSE SEARCH      │   │       SUPERPOWER 2: BM25 SEARCH       │
 │              (ChromaDB)               │   │           (Keyword Matcher)           │
 ├───────────────────────────────────────┤   ├───────────────────────────────────────┤
 │ Searches by MEANING.                  │   │ Searches by EXACT WORDS.              │
 │ Finds: "cancel" matches "terminate".  │   │ Finds: exact clause numbers ("14.2"), │
 │ Great for broad, conceptual queries.  │   │ product codes, monetary numbers.      │
 └───────────────────────────────────────┘   └───────────────────────────────────────┘
                     │                                           │
                     └───────────────────┬───────────────────────┘
                                         ▼
                     ┌───────────────────────────────────────┐
                     │           HYBRID RETRIEVER            │
                     │  (LangChain EnsembleRetriever + RRF)  │
                     │      50% Dense  +  50% BM25           │
                     └───────────────────────────────────────┘
                                         │
                                         ▼
                             BEST POSSIBLE RETRIEVAL!
```

### What is RRF? (Reciprocal Rank Fusion)
Imagine two judges at a dog show:
- Judge 1 (Dense) votes for Dog A, then Dog B.
- Judge 2 (BM25) votes for Dog B, then Dog C.
RRF combines both judges' rankings so that Dog B (which both judges liked) wins the prize!

---

## Part 9 — What Is LangGraph?

**Technical Term:** LangGraph  
**Simple Meaning:** The flowchart rules that guide the AI step-by-step.

LangGraph is **not** the LLM. LangGraph is the **conductor of the orchestra**. It tells the system what to do first, what to check next, and when to stop.

```
                                  START
                                    │
                                    ▼
                             [Retrieve Node]
                         (Fetches top 5 chunks)
                                    │
                                    ▼
                               [Grade Node]
                    (Did we find enough information?)
                                    │
                  ┌─────────────────┴─────────────────┐
             Sufficient = YES                    Sufficient = NO
                  │                                   │
                  │                            Rewrites < 2?
                  │                             ┌─────┴─────┐
                  │                           YES           NO
                  │                            │             │
                  │                            ▼             ▼
                  │                     [Rewrite Node]  (Low Confidence)
                  │                            │             │
                  │                            ▼             │
                  │                     [Retrieve Node]      │
                  │                            │             │
                  │                            ▼             │
                  │                       [Grade Node]       │
                  │                            │             │
                  └────────────────────────────┼─────────────┘
                                               │
                                               ▼
                                        [Generate Node]
                               (Writes answer with citations)
                                               │
                                               ▼
                                   [Hallucination Check Node]
                                 (Is answer strictly grounded?)
                                               │
                                               ▼
                                              END
```

---

## Part 10 — What Is Query Rewriting?

Sometimes humans ask questions that are short, messy, or vague.

**User asks:**  
*"What about walking away?"*

If we search a legal contract for *"walking away"*, the contract probably says nothing about walking! It talks about *"voluntary termination"*, *"breach of agreement"*, or *"cancellation clauses"*.

### What the Rewrite Node Does:
1. Detects that the initial search found poor clues.
2. Rewrites the search query using legal terminology:  
   *"What does the contract state regarding voluntary termination, cancellation rights, or early exit?"*
3. Searches the document again with the better query!

> **Code Rule:** The system is strictly bounded to a **maximum of 2 query rewrites**. It will never get stuck in an endless loop.

---

## Part 11 — What Is a Hallucination?

**Technical Term:** Hallucination  
**Simple Meaning:** When an AI invents "facts" that do not exist in the document!

### The Blank Space Example from Our Actual Test Contract:
In one of our test contracts, the contract text says:
> *"The monthly management service charge shall be US$ ________ per month."*  
> *(Notice that the line was left completely blank by the lawyers!)*

- ❌ **Hallucinated Answer:**  
  *"The monthly management service charge is $5,000 per month."*  
  *(The AI made up $5,000 out of thin air! That is a hallucination!)*

- ✔️ **Truthful, Grounded Answer:**  
  *"The exact monthly service charge is not specified in the document; the corresponding field is left blank."*

---

## Part 12 — What Is Grounding?

**Technical Term:** Grounding  
**Simple Meaning:** Being able to point your finger at the page and say: *"Look! It says it right here!"*

```
    ANSWER: "The Buyer must open a letter of credit within 30 days."
                               │
                               │  (Is there proof?)
                               ▼
    EVIDENCE: [Page 2, Clause 12.1]
    "The buyer shall open an irrevocable letter of credit within 30 days..."
                               │
                               ▼
                  ✔️ PROVEN GROUNDED FACT!
```

If an answer contains facts that cannot be pointed to on any retrieved page, **the grounding check fails**.

---

## Part 13 — What Is an Evaluation Dataset?

To test our assistant fairly, we need an **answer key**—just like a school teacher has an answer key when grading an exam.

Our dataset file is located at:  
📁 `evaluation/cuad_eval_dataset.json`

Each question entry contains:

```json
{
  "id": 1,
  "question": "What is specified regarding Agreement Date in the contract?",
  "expected_answer": "6th day of April, 1999",
  "contract_id": "cuad_contract_1.pdf",
  "clause_category": "Agreement Date",
  "ground_truth_context": "6th day of April, 1999",
  "is_out_of_domain": false
}
```

### The 5 Essential Parts of Evaluation:
1. **Document:** The contract being read (`cuad_contract_1.pdf`).
2. **Question:** What we ask the assistant.
3. **Reference Answer (Ground Truth):** What a human legal expert says the correct answer is.
4. **Retrieved Context:** The actual pieces our search engine found.
5. **Generated Answer:** The answer our assistant wrote.

---

## Part 14 — What Is CUAD?

**Technical Term:** CUAD  
**Full Name:** Contract Understanding Atticus Dataset  
**Simple Meaning:** A famous collection of real-world business contracts labeled by legal experts to test whether computers understand legal clauses.

### How OUR Project Uses CUAD:
In `evaluation/cuad_eval_dataset.json`, we have **45 carefully chosen evaluation questions**:
- **35 In-Domain Questions:** Covering 25+ real contract clause categories (Agreement Date, Parties, Effective Date, Termination Notice Period, Governing Law, Non-Compete, Renewal Terms, Payment Terms, Audit Rights, Insurance, etc.).
- **10 Out-of-Domain Refusal Questions:** Questions that have nothing to do with contracts (e.g., *"What is the capital of France?"*, *"How far is the Moon from the Earth?"*, *"Who won the 2022 World Cup?"*).

### Why Held-Out Evaluation Matters (The Student Exam Analogy)
If a teacher gives students the exact same 5 questions for homework, and then puts those exact same 5 questions on the final exam, the student might just memorize the words without understanding them!  
To test true understanding, we test on **unseen questions and separate contracts**!

---

## Part 15 — Retrieval Evaluation (Hit Rate@5)

Before we even look at the answer, we must ask:  
**"Did our search engine even find the page containing the answer?"**

### The Basket Analogy 🧺
Imagine you ask a child to fetch 5 apples from the orchard. You told them to make sure they get at least one golden apple.
- They come back with a basket of 5 apples.
- You look inside the basket.
- If the golden apple is anywhere in the basket of 5 → **It's a HIT! (1.0)**
- If the golden apple is not in the basket → **It's a MISS! (0.0)**

```
               QUESTION: "Who is the Buyer?"
               Correct answer is on Page 1.

   RETRIEVED 5 PIECES:
   [Page 1]  [Page 2]  [Page 4]  [Page 7]  [Page 9]
      ▲
      │
   FOUND! ──► HIT@5 = 1 (Success!)
```

### Mathematical Formula:
$$\text{Hit Rate@5} = \frac{\text{Number of Questions with Ground Truth in Top 5}}{\text{Total In-Domain Questions}}$$

If 34 out of 35 contract questions find the correct evidence in their top 5 results:
$$\text{Hit Rate@5} = \frac{34}{35} = 0.971 \quad (97.1\%)$$

---

## Part 16 — Why Retrieval Evaluation Is Crucial

```
   ┌───────────────────┐
   │   BAD RETRIEVAL   │ ──► Finds wrong pages ──► LLM gets bad clues ──► ❌ WRONG ANSWER
   └───────────────────┘

   ┌───────────────────┐
   │  GOOD RETRIEVAL   │ ──► Finds exact clause ──► LLM gets true facts ──► ✔️ ACCURATE ANSWER
   └───────────────────┘
```

If your retriever brings the wrong page, even the smartest AI in the world will fail. Evaluating retrieval separately helps us pinpoint whether a mistake was caused by the **search engine** or the **writer**!

---

## Part 17 — What Is Ragas?

**Technical Term:** Ragas  
**Simple Meaning:** An automated grading teacher designed specifically for RAG systems.

Instead of paying human lawyers to sit for weeks reading thousands of AI answers, Ragas provides standardized, mathematical formulas to measure every step of the pipeline.

In our project, we evaluate **4 core quality dimensions**:
1. **Faithfulness**
2. **Context Precision**
3. **Context Recall**
4. **Answer Relevancy**

Let's examine each one in detail!

---

## Part 18 — Faithfulness

### The Question It Asks:
> *"Did the answer stay 100% faithful to the evidence, or did it make things up?"*

### The Child Analogy 🎨
If the storybook says: *"The balloon was RED."*  
And the child says: *"The balloon was BLUE."*  
The child was **not faithful** to the storybook!

### How Our Code Actually Measures Faithfulness:
In [`app/evaluation/ragas_evaluator.py`](file:///e:/Project/capstone/app/evaluation/ragas_evaluator.py):
1. Takes the answer and splits it into individual sentences (claims).
2. Cleans out citation tags like `[Page 1]`.
3. For each sentence, extracts key content words, numbers, and dates (ignoring filler words like *the*, *and*, *is*).
4. Checks what percentage of those factual words appear directly in the retrieved context chunks!
5. If the LangGraph grounding node flagged any unsupported claims, it applies an explicit penalty.
6. If the question was an out-of-domain refusal, faithfulness to the document is correctly scored as `0.0`.

$$\text{Faithfulness} = \frac{\text{Supported Claims}}{\text{Total Claims in Answer}}$$

- **Score of 0.95 (95%):** Every statement in the answer is backed by the retrieved text.
- **Score of 0.40 (40%):** The answer is rambling or introducing outside information not found in the PDF.

---

## Part 19 — Context Precision

### The Question It Asks:
> *"Are the most useful clues ranked right at the TOP of the search results?"*

### The Snack Box Analogy 🥪
Imagine you are hungry for a sandwich. You open a lunchbox with 5 items:
- **Good Precision:** Item 1 is the sandwich! Item 2 is an apple.
- **Bad Precision:** Item 1 is a napkin, Item 2 is a spoon, Item 3 is a straw, Item 4 is a rubber band, and Item 5 at the very bottom is the sandwich.

Even though the sandwich was in the box, having it at the bottom wastes time and attention!

### How It Works:
Context Precision rewards systems that put the most relevant chunks at **Rank 1 and Rank 2**, rather than burying them at Rank 5.

---

## Part 20 — Context Recall

### The Question It Asks:
> *"Did the retriever find ALL the pieces of information needed to answer the question?"*

### The Puzzle Analogy 🧩
Imagine a puzzle that requires 2 pieces to see the picture:
- Piece A: *"The contract lasts for 1 year."*
- Piece B: *"Unless renewed by written notice 30 days prior."*

If the retriever only finds Piece A, but misses Piece B, the answer will be incomplete! That is a **recall failure**.

- **Precision = Quality:** Are the retrieved pieces relevant?
- **Recall = Quantity:** Did we get all the pieces we needed?

---

## Part 21 — Answer Relevancy

### The Question It Asks:
> *"Did the AI answer the actual question asked by the user?"*

### The Conversation Analogy 🗣️
- **User:** *"What time does the train arrive?"*
- **Good Answer:** *"The train arrives at 4:15 PM."* (High Relevancy: 1.0)
- **Bad Answer:** *"Trains are made of steel and run on tracks that were invented in the 19th century."* (Low Relevancy: 0.15 — true fact, but irrelevant to the user's question!)

Our system checks the keyword overlap and semantic intent between the question and the answer.

---

## Part 22 — Putting the Ragas Metrics Together

| Metric | What Part It Checks | Simple Question | Perfect Score |
|---|---|---|:---:|
| **Hit Rate@5** | Search Engine | *Did the proof make it into the top 5 results?* | 1.0 (100%) |
| **Context Recall** | Search Engine | *Did we find all needed clues from the reference?* | 1.0 (100%) |
| **Context Precision**| Search Ranking | *Are the best clues ranked near the top?* | 1.0 (100%) |
| **Faithfulness** | Answer Generator | *Is every claim in the answer truthful to the text?*| 1.0 (100%) |
| **Answer Relevancy** | Answer Generator | *Did the answer directly address the user's query?*| 1.0 (100%) |
| **Refusal Accuracy** | Guardrails | *Did the system refuse questions outside the PDF?* | 1.0 (100%) |

---

## Part 23 — What 90% Really Means (And What It Does NOT Mean)

> ⚠️ **CRITICAL DISTINCTION FOR SCIENTIFIC RIGOR**

When an evaluation report says:
$$\text{Faithfulness} = 0.92$$

It means:
> *"On the 45 test questions evaluated, approximately 92% of the factual claim tokens in the generated answers were directly substantiated by the retrieved context chunks."*

### What it does NOT mean:
- It does **not** mean *"The AI is 92% smart."*
- It does **not** mean *"The AI will be right on 92% of all future legal cases."*
- It does **not** mean *"A human lawyer is no longer needed."*

Evaluation metrics measure **specific mathematical properties**, not general human intelligence!

---

## Part 24 — Complete System & Evaluation Pipeline

```
+=============================================================================+
|                      FULL OFFLINE EVALUATION PIPELINE                       |
+=============================================================================+

                 evaluation/cuad_eval_dataset.json (45 Questions)
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
       35 In-Domain Questions                        10 Out-of-Domain Questions
   ("What are the payment terms?")                ("What is the capital of France?")
                 │                                             │
                 ▼                                             ▼
      [ Build In-Memory Index ]                     [ Fast Refusal Check ]
   (pdfplumber + all-MiniLM-L6-v2)                             │
                 │                                             ▼
                 ▼                                  Expected: Refusal
     ┌───────────────────────┐                                 │
     │   HYBRID RETRIEVAL    │                                 ▼
     │  ChromaDB  +   BM25   │                      Refusal Accuracy Checked!
     └───────────────────────┘
                 │
                 ├───► [ Measure Hit Rate@5 (Dense, BM25, Hybrid) ]
                 ├───► [ Measure Context Precision ]
                 └───► [ Measure Context Recall ]
                 │
                 ▼
     ┌───────────────────────┐
     │   LANGGRAPH REASON    │
     │  Grade -> Rewrite ->  │
     │  Generate -> Ground   │
     └───────────────────────┘
                 │
                 ▼
         Generated Answer
                 │
                 ├───► [ Calculate Faithfulness ]
                 └───► [ Calculate Answer Relevancy ]
                 │
                 ▼
     ┌─────────────────────────────────────────────────────────┐
     │                    OUTPUT REPORTS                       │
     │  1. evaluation/summary.json  (Aggregate benchmarks)    │
     │  2. evaluation/results.csv   (Row-by-row question data) │
     │  3. Terminal Formatted Table (Live viva presentation)   │
     └─────────────────────────────────────────────────────────┘
```

---

## Part 25 — Two Types of Testing

Our project has two distinct layers of defense:

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                     LAYER 1: RUNTIME PROTECTION                         │
 │                   (What protects the live user?)                        │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ • Vague question detector ("what?") ──► Asks for clarification          │
 │ • Out-of-domain filter ──► Immediately refuses general trivia           │
 │ • Self-correcting LangGraph loop ──► Rewrites failed search queries     │
 │ • Citation Validator ──► Verifies cited pages exist in retrieved chunks │
 │ • Live Answer Card ──► Displays real-time Relevance & Quality badge     │
 └─────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                     LAYER 2: OFFLINE EVALUATION                         │
 │                 (How do engineers measure quality?)                     │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ • evaluate.py script runs all 45 test questions against CUAD contracts  │
 │ • Automated Hit@5 comparison (Dense vs BM25 vs Hybrid)                  │
 │ • Automated Ragas computation (Faithfulness, Precision, Recall, Relev.) │
 │ • Saves detailed results to results.csv and summary.json                │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 26 — Out-of-Domain Evaluation (Saying "I Don't Know")

A safe AI must know what it does **not** know.

### Test Question:
> *"What is the capital of France?"*

- ❌ **Wrong Behavior:** *"The capital of France is Paris."*  
  *(Even though Paris is true in the real world, Paris is NOT in the uploaded contract! Giving this answer violates document boundaries!)*
- ✔️ **Correct Behavior:** *"This question is outside the scope of the uploaded document. I can only answer questions based on the document."*

In `evaluate.py`, we test 10 such questions. If the assistant correctly refuses all 10:
$$\text{Refusal Accuracy} = \frac{10}{10} = 1.0 \quad (100\%)$$

---

## Part 27 — Hallucination Evaluation (Testing Blank Spaces)

Lawyers often leave blank underline spaces in template contracts:
> *"Interest rate: ______% per annum."*

When asked: *"What is the interest rate?"*  
The model must explicitly report that the rate is omitted or blank, rather than assuming standard market rates like 5% or 10%.

---

## Part 28 — Walkthrough of a Good Evaluation Example

### Step 1: Input
- **Contract:** `cuad_contract_0.pdf`
- **Question:** *"What are the payment terms?"*
- **Ground Truth Context:** Clause 12: *"The buyer shall open an irrevocable letter of credit within 30 days of signing the contract..."*

### Step 2: Retrieval
- Hybrid Retriever pulls 5 chunks.
- Chunk 1 (Rank 1, Page 2) contains Clause 12.
- **Hit@5:** `1` (Success)
- **Context Precision:** `1.0` (Relevant chunk at Rank 1)
- **Context Recall:** `1.0` (All terms found)

### Step 3: Reasoning
- Grade node: `is_sufficient = True`.
- Generate node: Explains letter of credit, 30-day timeline, 80/20 payment split, citing `[Page 2]`.
- Grounding check: `is_grounded = True`.

### Step 4: Metric Output
- **Faithfulness:** `0.94` (All claims verified against context)
- **Answer Relevancy:** `0.88` (Directly answers payment inquiry)

---

## Part 29 — Walkthrough of a Bad / Failure Example

### Step 1: Input
- **Question:** *"What is the employee dress code?"*
- **Contract:** Commercial Equipment Supply Agreement (Contains no dress code!)

### Step 2: Retrieval & LangGraph Loop
1. **Retrieve:** Fetches 5 chunks about machinery and delivery.
2. **Grade:** Grade node returns `is_sufficient = False` (*"No dress code found"*).
3. **Rewrite #1:** Rewrites to *"employee uniform, attire, or workplace dress standards"*.
4. **Retrieve #2:** Still finds nothing about clothing.
5. **Grade #2:** Still `is_sufficient = False`.
6. **Rewrite #2:** Rewrites to *"conduct and operational rules for personnel"*.
7. **Retrieve #3:** Still no dress code.
8. **Loop Exit:** Max rewrites (2) reached.
9. **Generate:** Outputs low-confidence notice:  
   *"[Low Confidence] I could not find enough information in the provided document..."*

### Step 3: Evaluation Verdict
- System correctly recognized its own boundary and refused to invent a dress code!

---

## Part 30 — How to Read the Actual Evaluation Report

When you run:
```powershell
python evaluate.py
```

The system prints the benchmark results table:

```
============================================================
  EVALUATION RESULTS
============================================================
  Number of evaluation questions : 45
  Evaluation duration            : ~35.0s
  Results saved to               : evaluation/results.csv and evaluation/summary.json

  RETRIEVAL METRICS (In-Domain Hit@5)
  ----------------------------------------
  Dense Hit@5                    : 0.97
  BM25 Hit@5                     : 0.83
  Hybrid Hit@5                   : 0.97

  RAGAS GENERATION & CONTEXT METRICS
  ----------------------------------------
  Faithfulness                   : 0.89
  Context Precision              : 0.64
  Context Recall                 : 0.79
  Answer Relevancy               : 0.74
  Out-of-Domain Refusal Accuracy : 1.00

============================================================
  TARGET COMPARISON
============================================================
  Metric                    | Target     | Actual     | Status
  ----------------------------------------------------------
  Faithfulness              | >0.85      | 0.89       | MET
  Context Recall@5          | >0.75      | 0.79       | MET
  Refusal Accuracy          | >0.90      | 1.00       | MET
============================================================
```

> **Note:** The thresholds (`>0.85`, `>0.75`, `>0.90`) are **PROJECT TARGETS** established in our Phase 4 specifications, not universal laws of nature!

---

## Part 31 — Troubleshooting Guide: What Each Failure Means

| Metric That Failed | What Went Wrong Under the Hood | How an Engineer Fixes It |
|---|---|---|
| **Low Hit Rate@5** | The search engine is not finding the right pages. | Tune chunk sizes or test a stronger domain embedding model. |
| **Low Context Precision** | Good chunks are found, but buried under junk chunks. | Adjust BM25/Dense fusion weights or add a reranker model. |
| **Low Context Recall** | Chunks are too small; sentences get cut in half. | Increase child chunk size or increase chunk overlap. |
| **Low Faithfulness** | The AI is hallucinating or adding unmentioned facts. | Strengthen the generator prompt rules and lower temperature. |
| **Low Answer Relevancy** | The AI is answering with off-topic legal essays. | Improve prompt instruction to answer the exact question directly. |
| **Low Refusal Accuracy** | The AI answers general trivia instead of sticking to the PDF. | Strengthen the fast out-of-domain classifier keywords. |

---

## Part 32 — Why Not Just Ask a Human?

| Factor | Human Lawyer Evaluation | Automated Pipeline (`evaluate.py`) |
|---|---|---|
| **Speed for 45 questions** | 4 to 6 hours | 35 seconds |
| **Cost** | Very expensive ($150+/hr) | Effectively $0.00 (CPU + free tier) |
| **Consistency** | Mood & fatigue alter scoring | 100% deterministic & repeatable |
| **Code Regressions** | Hard to run after every git commit | Can run automatically in CI/CD tests |

*Humans are great for creating the reference questions once. Automated evaluation is great for testing the code every day!*

---

## Part 33 — Real-World Limitations

1. **Automated Evaluators Are Proxies:** Lexical and token overlap approximations are very close to human judgment, but they cannot replace a qualified lawyer in court.
2. **Quality of the Dataset:** If the human who wrote the reference answer made a typo, the computer will grade against that typo.
3. **Complex Cross-Page Synthesis:** If an answer requires combining 10 clauses across 80 pages, a top-5 retrieval limit may miss subtle edge conditions.
4. **OCR & Scanned PDFs:** If a PDF is a blurry photocopy with no text layer, text extraction requires an external OCR engine.

---

## Part 34 — One-Page Cheat Sheet

- **RAG:** Look up document facts first, then write the answer.
- **Embedding:** Turn words into numbers that capture meaning (`all-MiniLM-L6-v2`).
- **ChromaDB:** Temporary in-memory filing cabinet (RAM only, ephemeral).
- **BM25:** Search using exact important words.
- **Hybrid Retrieval:** Dense search + BM25 search combined with 50/50 RRF.
- **Top-K:** The top $K$ most relevant chunks (we use $K=5$).
- **LangGraph:** Flowchart controlling retrieve, grade, rewrite, generate, and verify steps.
- **Query Rewrite:** Rephrases vague user questions into legal search terms (Max 2 rewrites).
- **Hallucination:** When an AI invents facts not found in the document.
- **Grounding:** Every claim in the answer has proven page citations.
- **Hit Rate@5:** Percentage of questions where the true clue was found in the top 5 results.
- **Faithfulness:** Proportion of answer claims backed by the retrieved text.
- **Context Precision:** Whether the best clues are ranked at the top.
- **Context Recall:** Whether we retrieved enough clues to cover the reference answer.
- **Answer Relevancy:** Whether the answer directly addresses what the user asked.

---

## Part 35 — Explain It to a Child in 60 Seconds

> *"We give our computer a big contract.*  
> *The computer cuts the contract into small puzzle pieces.*  
> *It gives each piece a secret number code so it knows what each piece means.*  
>  
> *When someone asks a question, our computer searches for the best puzzle pieces.*  
> *Then a teacher robot checks: 'Do these pieces actually answer the question?'*  
> *If not, it tries asking the question in a better way.*  
>  
> *Then the AI writes a clean, simple answer using ONLY those puzzle pieces.*  
> *Finally, a checker robot proofreads the answer to make sure the AI didn't invent anything.*  
>  
> *After that, we run an automated test with 45 exam questions to measure:*  
> *1. Did it find the right pieces?*  
> *2. Did it tell the truth?*  
> *3. Did it stay on topic?*  
> *4. Did it say 'I don't know' to silly questions?*  
>  
> *That is our evaluation pipeline!"*

---

## Part 36 — Current Implementation vs Original Plan

| Feature Area | Original Conceptual Plan | Actual Current Implementation | Status |
|---|---|---|:---:|
| **PDF Ingestion** | PyPDF / pdfplumber | `pypdf` for text, `pdfplumber` + fallback for markdown tables | ✅ Implemented |
| **Chunking Architecture**| Single flat text chunking | **Parent/Child Chunking** (Parent: ~1000 chars, Child: ~800 chars, Overlap: 200 chars). Table docs are never split. | ✅ Implemented |
| **Vector Database** | Persistent ChromaDB on disk | **Ephemeral In-Memory ChromaDB** (`chromadb.EphemeralClient()`). Zero disk files, zero SQLite leakage. | ✅ Implemented |
| **Embeddings** | OpenAI text-embedding-ada-002 | `sentence-transformers/all-MiniLM-L6-v2` (Local CPU, fast, free, no keys needed). | ✅ Implemented |
| **Lexical Search** | Optional / Future | In-memory `BM25Retriever` (k=20). | ✅ Implemented |
| **Hybrid Retrieval** | Score addition | `EnsembleRetriever` with Reciprocal Rank Fusion (RRF), equal 50/50 weighting. | ✅ Implemented |
| **Reasoning Agent** | Unbounded loops or simple prompt | **LangGraph StateGraph** (retrieve → grade → conditional rewrite → generate → hallucination check). | ✅ Implemented |
| **Loop Constraint** | Max 3 rewrites | **Strictly maximum 2 rewrites** to prevent latency runaway. | ✅ Implemented |
| **LLM Provider** | OpenAI API only | **Groq SDK** (`gpt-oss-120b`) with **OpenAI SDK** (`gpt-4o-mini`) + deterministic offline fallback for CI/CD tests. | ✅ Implemented |
| **Evaluation Suite** | Manual evaluation | **Automated `evaluate.py`** with 45 CUAD questions, saving `results.csv` and `summary.json`. | ✅ Implemented |
| **Real-time Eval UI** | Planned only for offline batch | **Real-time Evaluation Card** in both FastAPI schema and Frontend UI showing Document Relevance %, Faithfulness, Alignment, and Relevancy per answer. | ✅ Implemented |
| **External Vector Cloud**| Pinecone / Qdrant | **Not implemented (By Design).** System strictly adheres to lightweight ephemeral in-memory privacy. | 🚫 Rejected by Design |

---

### Useful Commands Reference

```powershell
# 1. Run full offline evaluation pipeline (CUAD dataset + Hit@5 + Ragas):
python evaluate.py

# 2. Run automated test suite verifying guardrails, lifecycle, and API:
pytest tests/test_phase4.py tests/test_backend_api.py -v

# 3. Start the FastAPI backend server:
uvicorn backend.main:app --reload --port 8000
```
