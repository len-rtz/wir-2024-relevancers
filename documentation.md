# Web Information Retrieval - Winter Semester TH Köln 2024
## The Relevancers

### Introdcution
Information Retrieval (IR) is a subfield of computer science concerned with searching and retrieving information relevant to a user's query. 
It plays a role in various applications, including search engines, library catalogs, and recommender systems. The use case this project is taking into account is web search.

The core challenge in IR is bridging the gap between the user's intent expressed in a query and the information stored within a document collection. 
Among multiple other techniques query rewriting and semantic search address this challenge by reformulating or expanding the user's query to improve the retrieval process. 
These techniques aim to capture synonyms, related concepts, and semantic relationships that users might not explicitly include in their initial query.

This project is part of the Web Information Reteival course by Prof. Dr. Philipp Schaer at TH Köln. 

Within the context of the MS MARCO retrieval task, we will utilize a baseline retrieval system based on the BM25 ranking model and explore the potential for improvement through query rewriting with RM3, T5 model, and semantic search using sentence transformers.

The project will involve evaluating the effectiveness of RM3 query rewriting (our first baseline), T5 query rewriting, and semantic search compared to the overall baseline system (just BM25) and analyzing the impact on retrieval performance.

### Hypothesis
**H1**

The T5 query rewriter model (using [prhegde/t5-query-reformulation-RL](https://huggingface.co/prhegde/t5-query-reformulation-RL) will improve retrieval performance compared to both BM25 and BM25+RM3 baselines in terms of 
- mean average precision (MAP),
- normalized discounted cumulative gain (NDCG@10),
- and precision at various ranks (P@1, P@5, P@10).

**H2**

The semantic search approach (using [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) will show significant improvements over the baseline BM25 and BM25+RM3 in terms of 
- mean average precision (MAP),
- normalized discounted cumulative gain (NDCG@10),
- and precision at various ranks (P@1, P@5, P@10).

**H3**

Among all approaches, we expect the following performance ranking (from best to worst):
1. BM25 + T5
2. BM25 + Semantic Search
3. BM25 + RM3
4. BM25 (baseline)

**Why?**

**H1**

RM3 primarily relies on term frequencies and document similarities. As it primarily focuses on adding or removing terms from the original query. 
It may struggle to capture nuanced semantic relationships between words and concepts, leading to less effective query expansions.
Unlike rule-based methods like RM3, T5 should adapt to different query styles and domains. 
T5 can generate entirely new queries, explore different phrasings, and capture synonyms and related concepts.

**H2**

Dense vector representations should capture semantic relationships that term-based approaches miss while being more robust to vocabulary mismatch issues. It processes entire queries and documents holistically and maintains relationships between terms rather than treating them independently. Therefore it will be better at handling abstract concepts and thematic similarities.

**H3**

The T5 model we are using is specifically trained on query reformulation using MS MARCO data, which matches our task domain.
Semantic search helps identify conceptually similar documents. However, converting everything to dense vectors can sometimes lose fine-grained term importance. We assume that computing similarities in vector space is less interpretable than T5's explicit query reformulations
RM3 is a Proven technique that reliably provides incremental improvements but it only uses term statistics from the collection itself
it is limited to adding/removing terms rather than understanding deeper semantics.

### Basline System (BM25)
BM25 is a probabilistic retrieval model that ranks documents based on their relevance to a query. It considers factors like term frequency, document length, and the overall frequency of the term in the collection.
The baseline system uses only the BM25 algorithm.

This is the overall project's baseline system.

### Data Preprocessing
Before implementing any of our retrieval approaches, we established a basic preprocessing pipeline to ensure consistent and clean data.
The same steps were applied in all three systems.

The preprocessing steps include:
1. Text Cleaning:
   - Case normalization (conversion to lowercase)
   - Special character removal
   - Stopword removal using NLTK's English stopwords
   - Whitespace normalization
  
2. Indexer Configuration:
   - document iteration
   - Metadata handling for document IDs and text
   - Clean text storage with configurable field lengths

### "Our" Baseline System (BM 25 with RM3)
This section describes the BM25 retrieval model with relevant feedback using Rocchio (RM3) for query expansion, which builds upon the baseline BM25 system.
The system first performs retrieval using the BM25 ranking model. This retrieves a set of documents based on their relevance to the original user query.
The retrieved documents are assumed to be relevant to the user's information needs.
Terms from these documents are used to reformulate the query.
High-frequency terms from the relevant documents are added to the original query.
Low-frequency terms from the original query might be down-weighted or removed.
The reformulated query is used to retrieve documents again. 
This is integrated with the BM25 retrieval pipeline using the pt.rewrite.RM3(index) operator in PyTerrier.

This is our own baseline system.

### Test System H1 (BM 25 with T5 Query Rewriter)
Our first test system uses a Large Language Model for Query Rewriting.
In general T5's architecture is specifically designed for text-to-text generation tasks, making it well-suited for reformulating queries. 
It can capture the semantics and context of the original query and generate alternative phrasings that better represent the user's intent.

We are using the [prhegde/t5-query-reformulation-RL](https://huggingface.co/prhegde/t5-query-reformulation-RL)-model, a sequence-to-sequence model initialized with Google's t5-base model. 
It is trained using ms-marco query pairs data and fine-tuned with a Reinforcement Learning framework using a policy gradient approach to fine-tune the policy (sequence-to-sequence model).
For a given input query, a set of reformulated queries is sampled from the model, and reward is computed. Model updates are performed by applying a policy gradient algorithm.
The reward function is used to increase the model's capability to paraphrase. 

More information can be found [here](https://github.com/PraveenSH/RL-Query-Reformulation)

The T5 query rewriting system consists of two main components:

1. Core Query Rewriting Component (T5QueryRewriter):
  - Uses the pre-trained T5 model fine-tuned for query reformulation
  - Generates multiple reformulations for each query
  - Includes additional query cleaning and preprocessing
  - Integrates with PyTerrier's transformer framework


2. PyTerrier Integration Component:
  - Custom transformer class integrating with PyTerrier framework
  - Pipeline integration with BM25
  - Optional combination with RM3 expansion

### Test System H2 (BM25 with Semantic Search)
Our second test system uses a sentence-transformer model, primarily designed for semantic similarity tasks and embedding generation
We are using the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model from the Sentence Transformers library. This approach differs from traditional term-based retrieval methods by operating in a dense vector space that captures semantic relationships.

The semantic search system consists of two main components:

1. Core Semantic Search Component:
  - Uses the SentenceTransformer model 'all-MiniLM-L6-v2' for creating embeddings
  - Converts both documents and queries into dense vector representations
  - Employs cosine similarity for ranking documents
  - Retrieves top-k most similar documents for each query

2. PyTerrier Integration Component (SemanticSearchWrapper):
  - Custom transformer class integrating with PyTerrier framework
  - Handles batch processing of queries
  - Maintains compatibility with PyTerrier's evaluation pipeline
  - Returns results in standard PyTerrier format

### Evaluation
### Results 
The following table summarizes the performance of the different retrieval systems across the evaluation metrics:

| **System Name**          | **MAP**   | **Reciprocal Rank** | **NDCG@10** | **P@1**   | **P@5**   | **P@10**  |
|--------------------------|-----------|---------------------|-------------|-----------|-----------|-----------|
| **BM25 + RM3**           | 0.452199  | 0.768722            | 0.512417    | 0.680412  | 0.653608  | 0.612371  |
| **BM25 + Reform (T5)**   | 0.382817  | 0.697691            | 0.426565    | 0.587629  | 0.534021  | 0.508247  |
| **BM25 + Reform + RM3**  | 0.421044  | 0.696987            | 0.449395    | 0.608247  | 0.583505  | 0.545361  |
| **Semantic Search**      | 0.531523  | -                   | 0.652962    | 0.876289  | 0.793814  | 0.737113  |

#### Key Findings:
- **Semantic Search (Sentence-BERT)** achieved the **best performance** across all metrics, particularly excelling in **NDCG@10**, **P@1**, and **P@10**, showing its ability to handle **semantically rich** queries.
- **BM25 + T5** performed **better than BM25 + RM3** and **BM25 + Reform + RM3**, with a **significant improvement** in **MAP** and **P@1**. The T5-based query rewriting model demonstrates its ability to **generate diverse and context-aware rewrites**, leading to **better retrieval quality**.
- **BM25 + RM3** performed **well** in terms of **traditional metrics** like **precision**, but **lacked** the **semantic depth** of the other models, especially when dealing with more complex queries.
- **BM25 + Reform (T5)** showed **worse performance** than BM25 + RM3, which could be due to **ineffective fine-tuning** or **misalignment** in reformulating queries for the retrieval task.

### Limitations & Discussion
### Conclusion

