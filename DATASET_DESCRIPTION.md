# PRM Dataset Description

## Overview
This dataset contains interleaving A/B testing data with 30 items per request. Each line represents a list of items for each request, with various features and labels.

Source: https://github.com/hf4Academic/PRM?tab=readme-ov-file

## Data Structure
Each line in the dataset contains 6 fields separated by `|`:

1. **Request ID**: Unique identifier for each request
2. **User Category Features**: Category features of the user
3. **Item Category Features**: Category features of each item in the list
4. **Basic Dense Features**: Important dense features for each item
5. **Optional Dense Features**: Additional dense features for each item
6. **Labels**: Labels for each item (1.0 for positive, 0.0 for negative)

## Field Details

### 1. Request ID
- Type: Integer
- Example: `362730`

### 2. User Category Features
- Type: Array of 3 numerical values
- Example: `[2, 1, 3]`
- Represents category features of the user

### 3. Item Category Features
- Type: Array of 30 items, each with 5 numerical values
- Example: `[[5446604, 12, 2, 1, 1], [5744528, 3, 1, 5, 1], ...]`
- Represents category features of each item in the list

### 4. Basic Dense Features
- Type: Array of 30 arrays, each with 7 numerical values
- Example: `[[1.718472, 0.055282, 1.427542, 2.069278, 1.5304, 1.45515, 0.857918], ...]`
- Represents the important (basic) dense features for each item

### 5. Optional Dense Features
- Type: Array of 30 arrays, each with 12 numerical values
- Example: `[[-0.11826, 2.718945, 0.085299, -0.735397, 0.078699, 0.23584, 1.984058, 0.842391, 0.844605, 1.62474, 0.466446, -0.147121], ...]`
- Represents additional (optional) dense features for each item

### 6. Labels
- Type: Array of 30 numerical values (0.0 or 1.0)
- Example: `[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ...]`
- 1.0 represents positive labels, 0.0 represents negative labels

## Dataset Statistics
- Total requests: 6,707,158
- Items per request: 30
- Total items: 201,214,740

## Validation from Parquet Files
Verified with Parquet data structure:
- Each Parquet file contains 10,000 records (except the last one)
- All records have consistent array lengths:
  - Items features: 30 items per record
  - Basic dense features: 30 arrays of 7 values each
  - Optional dense features: 30 arrays of 12 values each
  - Labels: 30 values per record
- Data types preserved correctly in Parquet format