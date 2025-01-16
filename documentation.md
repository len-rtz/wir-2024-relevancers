# Web Information Retrieval - Winter Semester TH Köln 2024
## The Relevancers

### Introdcution
Information Retrieval (IR) is a subfield of computer science concerned with searching and retrieving information relevant to a user's query. 
It plays a role in various applications, including search engines, library catalogs, and recommender systems.

A core challenge in IR is bridging the gap between the user's intent expressed in a query and the information stored within a document collection. 
Among multiple other techniques Query rewriting addresses this challenge by reformulating or expanding the user's query to improve the retrieval process. 
These techniques aim to capture synonyms, related concepts, and alternative phrasings that users might not explicitly include in their initial query.

This project is part of the XY course. 
Within the context of the MS MARCO passage retrieval task we will utilize a baseline retrieval system based on the BM25 ranking model and explore the potential for improvement through query rewriting with RM3 and T5 model. 
The project will involve evaluating the effectiveness of RM3 query rewriting (our basline), T5 query rewriting compared to the overall baseline system (just BM25) and analyzing the impact on retrieval performance.

### Hypothesis
The T5 query rewriter model (using [prhegde/t5-query-reformulation-RL](https://huggingface.co/prhegde/t5-query-reformulation-RL) will improve retrieval performance compared to both BM25 and BM25+RM3 baselines in terms of 
- mean average precision (MAP),
- normalized discounted cumulative gain (NDCG@10),
- and precision at various ranks (P@1, P@5, P@10).

The observed improvements will be statistically significant at a confidence level of 30%.

**Why?**
RM3 primarily relies on term frequencies and document similarities. As it primarily focuses on adding or removing terms from the original query. 
It may struggle to capture nuanced semantic relationships between words and concepts, leading to less effective query expansions.
Unlike rule-based methods like RM3, T5 should adapt to different query styles and domains. 
T5 can generate entirely new queries, explore different phrasings, and capture synonyms and related concepts.

### Basline System (BM25)
BM25 is a probabilistic retrieval model that ranks documents based on their relevance to a query. It considers factors like term frequency, document length, and the overall frequency of the term in the collection.
The baseline system uses only the BM25 algorithm.

This is the overall project's baseline system.

### Baseline System (BM 25 with RM3)
This section describes the BM25 retrieval model with Relevance Feedback using Rocchio (RM3) for query expansion, which builds upon the baseline BM25 system.
The system first performs retrieval using the BM25 ranking model. This retrieves a set of documents based on their relevance to the original user query.
The retrieved documents are assumed to be relevant to the user's information need.
Terms from these documents are used to reformulate the query.
High-frequency terms from the relevant documents are added to the original query.
Low-frequency terms from the original query might be down-weighted or removed.
The reformulated query is used to retrieve documents again. 
This is integrated in the BM25 retrieval pipeline using the pt.rewrite.RM3(index) operator in PyTerrier.

This is our own baseline system.

### Test System (BM 25 with T5 Query Rewriter)
In general T5's architecture is specifically designed for text-to-text generation tasks, making it well-suited for reformulating queries. 
It can capture the semantics and context of the original query and generate alternative phrasings that better represent the user's intent.

We are using the [prhegde/t5-query-reformulation-RL](https://huggingface.co/prhegde/t5-query-reformulation-RL)-model, a sequence-to-sequence model initialized with Google's t5-base model. 
It is first trained supervised using ms-marco query pairs data. Then it's fine tuned weith a Reinforcement Learning framework using a policy gradient approach to fine-tune the policy (sequence-to-sequence model).
For a given input query, a set of reformulated queries are sampled from the model and reward is computed. Model updates are performed by applying a policy gradient algorithm.
The reward function is used to increases the model's capability to paraphrase. 

More information can be found [here](https://github.com/PraveenSH/RL-Query-Reformulation)
### Evaluation
### Results 
### Limitations & Discussion
### Conclusion

