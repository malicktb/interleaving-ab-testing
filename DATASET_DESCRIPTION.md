# PRM Dataset Description

Source: https://github.com/hf4Academic/PRM?tab=readme-ov-file

## What the Data Looks Like

Here's an actual example from the dataset:

**Request ID: 442358**

**User Context:**
```
[4.0, 1.0, 1.0]  ← 3 category features describing this user
```

**30 Items shown to the user (showing first 2):**

Item 0:
- Category Features: `[2659831, 3, 1, 5, 1]` ← Item ID and category codes
- Basic Dense: `[1.770, -0.387, -1.615, -1.166, -2.010, 1.476, -2.327]` ← 7 features
- Optional Dense: `[1.448, -0.263, 0.199, ...]` ← 12 more features
- **Label: 1.0** ← User clicked/interacted with this item

Item 1:
- Category Features: `[6012792, 29, 1, 1, 1]`
- Basic Dense: `[1.651, 0.551, 1.781, 1.573, 1.002, 1.427, 1.239]`
- Optional Dense: `[0.634, -0.421, -0.302, ...]`
- **Label: 0.0** ← No interaction

... (28 more items)

## Data Structure
Each record contains 6 fields:

1. **Request ID**: Unique identifier for each request
2. **User Category Features**: 3 category values describing the user
3. **Item Category Features**: For each of 30 items, 5 category values (likely item ID + categories)
4. **Basic Dense Features**: For each of 30 items, 7 normalized numerical features
5. **Optional Dense Features**: For each of 30 items, 12 additional normalized features
6. **Labels**: For each of 30 items, binary label (1.0 = interaction, 0.0 = no interaction)

## Field Details

### 1. Request ID
- **Type:** Integer
- **Example:** `442358`
- Unique identifier for this request

### 2. User Category Features
- **Type:** Array of 3 floats
- **Example:** `[4.0, 1.0, 1.0]`
- Category features describing the user making the request

### 3. Item Category Features
- **Type:** 30 arrays, each with 5 integers
- **Example:** `[[2659831, 3, 1, 5, 1], [6012792, 29, 1, 1, 1], ...]`
- First value appears to be item ID, remaining 4 are likely category/subcategory codes

### 4. Basic Dense Features
- **Type:** 30 arrays, each with 7 floats
- **Example:** `[[1.770, -0.387, -1.615, ...], [1.651, 0.551, 1.781, ...], ...]`
- Normalized numerical features (range approximately -3 to 3)
- Core features for ranking/recommendation

### 5. Optional Dense Features
- **Type:** 30 arrays, each with 12 floats
- **Example:** `[[1.448, -0.263, 0.199, ...], [0.634, -0.421, -0.302, ...], ...]`
- Additional normalized numerical features
- Supplementary features for model training

### 6. Labels
- **Type:** Array of 30 floats (0.0 or 1.0)
- **Example:** `[1.0, 0.0, 0.0, 0.0, 0.0, ...]`
- **1.0** = user interacted with this item (clicked, purchased, etc.)
- **0.0** = no interaction
- Typically sparse (most items have 0.0)

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
- **Structure:** All array lengths are consistent across records
  - User features: Always 3 values
  - Items per record: Always 30
  - Item category features: Always 5 per item
  - Basic dense features: Always 7 per item
  - Optional dense features: Always 12 per item
  - Labels: Always 30 per record
