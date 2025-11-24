# PRM Dataset Description

## Overview

This dataset comes from the PRM (Personalized Re-ranking for Recommendation) research from Alibaba's Taobao e-commerce platform. It contains real production data for training learning-to-rank and recommendation models.

This dataset contains full lists of items shown to real users along with the corresponding user clicks/interactions. This enables offline simulation of recommendation experiments without needing to interact with live users - you can evaluate different ranking algorithms against actual user behavior.

**Source:** https://github.com/hf4Academic/PRM

## Important: Feature Documentation Limitations

**The feature semantics in this dataset are intentionally undocumented.** This is a proprietary industrial dataset where:

- Feature meanings are not disclosed (business-sensitive information)
- Researchers are expected to treat features as opaque numerical/categorical inputs
- The focus is on re-ranking methodology, not feature engineering
- This is standard practice for anonymized production datasets

**What IS documented:** Data structure, types, dimensions, and formats
**What is NOT documented:** Semantic meanings of individual feature dimensions

For details on the re-ranking methodology, see: [Personalized Re-ranking for Recommendation (RecSys 2019)](https://arxiv.org/abs/1904.06813)

## What the Data Looks Like

Here's an actual example from the dataset:

**Request ID: 442358**

**User Context:**
```
[4.0, 1.0, 1.0]  ← 3 category features describing this user
```

**30 Items shown to the user (showing first 2):**

Item 0:
- Category Features: `[2659831, 3, 1, 5, 1]` ← 5 categorical values (meanings not documented)
- Basic Dense: `[1.770, -0.387, -1.615, -1.166, -2.010, 1.476, -2.327]` ← 7 normalized features
- Optional Dense: `[1.448, -0.263, 0.199, ...]` ← 12 additional features
- **Label: 1.0** ← User clicked/interacted with this item

Item 1:
- Category Features: `[6012792, 29, 1, 1, 1]` ← Different categorical values
- Basic Dense: `[1.651, 0.551, 1.781, 1.573, 1.002, 1.427, 1.239]`
- Optional Dense: `[0.634, -0.421, -0.302, ...]`
- **Label: 0.0** ← No interaction

... (28 more items)

## Data Structure
Each record contains 6 fields:

1. **Request ID**: Unique identifier for each request
2. **User Category Features**: 3 categorical values (semantics not specified in source)
3. **Item Category Features**: For each of 30 items, 5 categorical values (semantics not specified)
4. **Basic Dense Features**: For each of 30 items, 7 normalized numerical features (semantics not specified)
5. **Optional Dense Features**: For each of 30 items, 12 additional normalized features (semantics not specified)
6. **Labels**: For each of 30 items, binary label (1.0 = interaction, 0.0 = no interaction)

## Field Details

### 1. Request ID
- **Type:** Integer
- **Example:** `442358`
- Unique identifier for this request

### 2. User Category Features
- **Type:** Array of 3 floats
- **Example:** `[4.0, 1.0, 1.0]`
- **Semantics:** Not documented in source material
- Could represent user demographics, segments, behavior categories, or other user attributes
- Treat as opaque categorical inputs for model training

### 3. Item Category Features
- **Type:** 30 arrays, each with 5 integers
- **Example:** `[[2659831, 3, 1, 5, 1], [6012792, 29, 1, 1, 1], ...]`
- **Semantics:** Not documented in source material
- Could represent item IDs, product categories, brand IDs, or other item attributes
- The first value is a large integer (possibly item ID), remaining 4 are smaller integers (possibly category hierarchy)
- Treat as opaque categorical inputs for model training

### 4. Basic Dense Features
- **Type:** 30 arrays, each with 7 floats
- **Example:** `[[1.770, -0.387, -1.615, ...], [1.651, 0.551, 1.781, ...], ...]`
- **Semantics:** Not documented in source material
- Normalized/standardized numerical features (range approximately -3 to 3)
- Described in source as "important features which are suggested to be used"
- Could represent CTR predictions, popularity scores, price features, quality metrics, or other signals
- Treat as core opaque numerical inputs for ranking models

### 5. Optional Dense Features
- **Type:** 30 arrays, each with 12 floats
- **Example:** `[[1.448, -0.263, 0.199, ...], [0.634, -0.421, -0.302, ...], ...]`
- **Semantics:** Not documented in source material
- Additional normalized numerical features
- Described in source as "supplementary" or "optional" features
- Could represent additional engagement metrics, temporal features, or contextual signals
- Treat as supplementary opaque numerical inputs for model training

### 6. Labels
- **Type:** Array of 30 floats (0.0 or 1.0)
- **Example:** `[1.0, 0.0, 0.0, 0.0, 0.0, ...]`
- **Semantics:** Well-documented - binary interaction labels
- **1.0** = positive interaction (user clicked, purchased, or engaged with this item)
- **0.0** = no interaction
- Typically sparse (most items have 0.0 since users interact with few items)
- This is the supervised signal for training ranking models

## Dataset Statistics
- **Total requests:** 6,707,158
- **Items per request:** 30 (consistent across all records)
- **Total items:** 201,214,740
- **Parquet chunks:** 671 files
- **Records per chunk:** 10,000 (except last chunk)

## Storage Format

The dataset is stored in Parquet format for efficient access:
- **Location:** `s3://amzn-dataset-bucket/parquet_chunks/`
- **Access:** Public read (no credentials required)
- **Compression:** Snappy
- **Structure:** All array lengths are consistent across records
  - User features: Always 3 values
  - Items per record: Always 30
  - Item category features: Always 5 per item
  - Basic dense features: Always 7 per item
  - Optional dense features: Always 12 per item
  - Labels: Always 30 per record

## References

- **Original Paper:** Pei et al., "Personalized Re-ranking for Recommendation", RecSys 2019
  - Paper: https://arxiv.org/abs/1904.06813
  - ACM: https://dl.acm.org/doi/10.1145/3298689.3347000
- **Source Repository:** https://github.com/hf4Academic/PRM
- **Author Homepage:** https://hf4academic.github.io/

## Notes on Feature Engineering

Since feature semantics are not provided, researchers typically:
- Use features as-is without manual feature engineering
- Apply embedding layers to categorical features
- Use raw numerical features as model inputs
- Focus on model architecture rather than feature interpretation
- Treat this as a benchmark for evaluating ranking algorithms, not for feature analysis
