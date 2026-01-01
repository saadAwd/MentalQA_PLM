# Manual Evaluation Guide

## Overview

A manual evaluation tab has been added to the Gradio app for evaluating retrieval test results. This allows you to rate the relevance of retrieved answers on a scale of 1-5 and add comments.

## Features

1. **Rating System (1-5)**:
   - **5**: Retrieved answer is directly relevant and provides a solution
   - **4**: Retrieved answer is mostly relevant with minor gaps
   - **3**: Retrieved answer is somewhat relevant but incomplete
   - **2**: Retrieved answer has limited relevance
   - **1**: Retrieved chunks are not relevant

2. **Comment Section**: Add any related comments about the evaluation

3. **Navigation**:
   - First/Previous/Next/Last buttons
   - Jump to specific index
   - Fast access indicators showing completed evaluations

4. **Auto-save**: Evaluations are saved automatically when you:
   - Click "Save Evaluation"
   - Click "Next" (saves and moves forward)
   - Click "Previous" (saves current and moves backward)

5. **Statistics**: Shows progress (evaluated/total, percentage complete)

6. **Fast Access**: Visual indicators showing which questions have been evaluated (color-coded by rating)

## Usage

1. Launch the Gradio app:
   ```bash
   python -m knowldege_base.rag_staging.gradio_retriever_test
   ```

2. Navigate to the "📝 Manual Evaluation" tab

3. Review the displayed information:
   - ID
   - Question
   - Original Answer
   - Retrieved Answer
   - Scores (alpha, hybrid, BM25, MARBERT, Reranker)

4. Rate the relevance using the slider (1-5)

5. Optionally add comments

6. Click "Save Evaluation" or use Next/Previous to save and navigate

7. Use the "Jump to Index" field to quickly navigate to a specific question

## Data Storage

Evaluations are saved to `retrieval_test_results.json` with the following fields added:
- `manual_rating`: Integer from 1-5
- `manual_comment`: String with your comments

## Files

- `manual_evaluation.py`: Core evaluation module
- `knowldege_base/rag_staging/gradio_retriever_test.py`: Modified to include the manual evaluation tab

## Notes

- The evaluation data is saved in the same JSON file as the retrieval results
- You can resume evaluation at any time - your previous ratings and comments are preserved
- The fast access section shows completed evaluations (first 100) with color coding:
  - Red (1): Not relevant
  - Orange (2): Limited relevance
  - Yellow (3): Somewhat relevant
  - Light Green (4): Mostly relevant
  - Green (5): Directly relevant

