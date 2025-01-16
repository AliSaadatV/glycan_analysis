# Reflection on Project

## Time Spent on Each Section
- **Task 1:** Took approximately 4 days – 2 days for literature review and learning best practices in the field, and 2 days for coding.
- **Task 2:** Also took about 4 days – 2 days for setting up the code base for training and 2 days for analyzing the trained model and embeddings.

## Perceived Difficulty of Different Components
Both tasks were manageable overall. However, **Task 1** required more extensive reading to become familiar with the field, understand the terminology, and adopt best practices and recommendations.

## What Worked Well
- For both tasks, the current solutions are solid but have room for improvement:
  - **Task 1:** Guidance from an experimental expert with deep knowledge of glycans could enhance filtering and batch/blank correction processes.
  - **Task 2:** The initial model used a BERT architecture for masked-language-modeling (MLM) with default hyperparameters. There’s potential to fine-tune hyperparameters to optimize performance.

## Challenges
- **Task 1:** Data processing and discriminatory analysis required careful attention. Previous publications, such as [this paper](https://www.nature.com/articles/nprot.2011.335) and [this one](https://pmc.ncbi.nlm.nih.gov/articles/PMC5960010/), served as references.
- **Task 2:** Tokenization was challenging. Improving the biological relevance of glycan tokenization is key. A WordPiece tokenizer was trained and performed well, but further refinements are possible to enhance its effectiveness.


## Future Steps
- **Task 1:** Engage further with experimental experts to refine the analysis and validate the findings. Their insights could help identify key areas for improvement and ensure the results are robust and meaningful.

- **Task 2:** The application of novel AI/ML techniques in glycomics remains largely unexplored, presenting a significant opportunity to develop state-of-the-art solutions. This can be achieved by systematically investigating the following aspects:
  1. **Representation**: Glycans can be represented as sequences or graphs, each offering unique advantages depending on the analysis context.
  2. **Model Architecture**: Explore a variety of architectures such as transformers, Mamba, xLSTM, Hyena, CNNs, and others to identify the most effective approach for glycomics data.
  3. **Training Scenarios**:
     - **Self-supervised Learning**: Techniques like masked language modeling, next token prediction, or contrastive learning.
     - **Supervised Learning**: For tasks such as multilabel prediction.
     - **Hybrid Approaches**: Combine self-supervised pretraining with supervised fine-tuning (e.g., pretrain using self-supervised methods, then fine-tune for multilabel prediction tasks).

