# Reflection on Project

## Time Spent on Each Section
- **Task 1:** Took approximately 4 days – 2 days for literature review and learning best practices in the field, and 2 days for coding.
- **Task 2:** Also took about 4 days – 2 days for setting up the code base for training and 2 days for analyzing the trained model and embeddings.

## Perceived Difficulty of Different Components
Both tasks were manageable overall. However, **Task 1** required more extensive reading to become familiar with the field, understand the terminology, and adopt best practices and recommendations.

## What Worked Well
- For both tasks, the current solutions are solid but have room for improvement:
  - **Task 1:** Guidance from an experimental expert with deep knowledge of glycans could enhance filtering and batch/blank correction processes.
  - **Task 2:** The initial model used a BERT architecture with default hyperparameters. There’s potential to explore newer architectures (e.g., RoBERTa or ModernBERT) and fine-tune hyperparameters to optimize performance.

## Challenges
- **Task 1:** Data processing and discriminatory analysis required careful attention. Previous publications, such as [this paper](https://www.nature.com/articles/nprot.2011.335) and [this one](https://pmc.ncbi.nlm.nih.gov/articles/PMC5960010/), served as references.
- **Task 2:** Tokenization was challenging. Improving the biological relevance of glycan tokenization is key. A WordPiece tokenizer was trained and performed well, but further refinements are possible to enhance its effectiveness.
