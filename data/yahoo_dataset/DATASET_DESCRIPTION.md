# Yahoo! Learning to Rank Challenge Dataset

## Overview

The C14 Yahoo! Learning to Rank Challenge dataset (version 1.0) is designed to benchmark machine learning algorithms for web search ranking. It consists of features extracted from query-URL pairs along with relevance judgments. Query text, URLs, and feature descriptions are anonymized; only feature values are provided.

https://huggingface.co/datasets/YahooResearch/Yahoo-Learning-to-Rank-Challenge

## Dataset Statistics

### Summary

| Metric | Value |
|--------|-------|
| Total Documents | 882,747 |
| Total Queries | 36,251 |
| Feature Dimension | 700 |
| Relevance Levels | 5 (0-4) |

### Set 1 (Primary Dataset)

| Split | Queries | Documents | Docs/Query (avg) |
|-------|---------|-----------|------------------|
| Train | 19,944 | 473,134 | 23.7 |
| Valid | 2,994 | 71,083 | 23.7 |
| Test | 6,983 | 165,660 | 23.7 |
| **Total** | **29,921** | **709,877** | |

### Set 2

| Split | Queries | Documents | Docs/Query (avg) |
|-------|---------|-----------|------------------|
| Train | 1,266 | 34,815 | 27.5 |
| Valid | 1,266 | 34,881 | 27.6 |
| Test | 3,798 | 103,174 | 27.2 |
| **Total** | **6,330** | **172,870** | |

## Feature Information

| Metric | Set 1 | Set 2 | Combined |
|--------|-------|-------|----------|
| Feature Count | 519 | 595 | 699 (union) |
| Feature ID Range | 1-699 | 1-700 | 1-700 |
| Intersection | - | - | 415 |
| Avg Features/Doc | 223 | 229 | - |
| Sparsity | 68.1% | 67.3% | - |

All features are normalized to the [0, 1] range.

## Relevance Distribution

| Grade | Count | Percentage | Description |
|-------|-------|------------|-------------|
| 0 | 223,089 | 25.3% | Not relevant |
| 1 | 340,921 | 38.6% | Marginally relevant |
| 2 | 241,265 | 27.3% | Fairly relevant |
| 3 | 61,172 | 6.9% | Highly relevant |
| 4 | 16,300 | 1.8% | Perfectly relevant |

## File Format

### Source Format (LETOR)

```
<relevance> qid:<query_id> <fid>:<value> <fid>:<value> ...
```

Example:
```
3 qid:1 1:0.5 5:0.8 10:0.3 ...
```

### Parquet Format (Converted)

| Column | Type | Description |
|--------|------|-------------|
| query_id | string | Query identifier |
| features | list[list[float]] | Dense feature matrix (N x 700) |
| relevance | list[int] | Relevance grades for N documents |
