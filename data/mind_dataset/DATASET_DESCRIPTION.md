# MIND Dataset Description

## Overview

The Microsoft News Dataset (MIND) is a large-scale benchmark dataset for news recommendation research. It was collected from anonymized behavior logs of Microsoft News (MSN News) website.

**Source:** https://msnews.github.io/

**Paper:** Wu et al., "MIND: A Large-scale Dataset for News Recommendation", ACL 2020
- arXiv: https://arxiv.org/abs/2010.12836
- ACL Anthology: https://aclanthology.org/2020.acl-main.331/

## What the Data Looks Like

Here's an actual example from the dataset:

**Impression ID: 1**

**User:** U87243
**Timestamp:** 11/10/2019 11:30:54 AM

**User History (16 previously clicked articles):**
```
N8668 N39081 N65259 N79529 N73408 N43615 N29379 N32031 N110232 N101921 N12614 N129591 N105760 N60457 N1229 N64932
```

**Impressions shown (19 articles with click labels):**
```
N78206-0 N26368-0 N7578-0 N58592-0 N19858-0 N58258-0 N18478-0 N2591-0 N97778-0 N32954-0 N94157-1 N39404-0 N108809-0 N78699-1 N71090-1 N40282-0 N31174-1 N37924-0 N27822-0
```
- Format: `NewsID-Label` where Label: 1=clicked, 0=not clicked
- In this example, user clicked 4 articles: N94157, N78699, N71090, N31174

**Sample News Article (N1):**
```
ID: N1
Category: sports
Subcategory: football_nfl
Title: Texans defensive tackle D.J. Reader is taking advantage of his opportunities
Abstract: Houston Texans defensive tackle D.J. Reader is taking advantage of opportunities given by defensive end J.J. Watt.
Entities: D.J. Reader (Person, Q24007178), Houston Texans (Organization, Q223514), J.J. Watt (Person, Q1097511)
```

## Data Files

### 1. behaviors.tsv

User impression logs recording what news was shown and clicked.

| Column | Name | Type | Description |
|--------|------|------|-------------|
| 1 | Impression ID | Integer | Unique identifier for each impression/session |
| 2 | User ID | String | Anonymized user identifier (e.g., U87243) |
| 3 | Time | String | Timestamp: "MM/DD/YYYY HH:MM:SS AM/PM" |
| 4 | History | String | Space-separated news IDs the user clicked before this session |
| 5 | Impressions | String | Space-separated "NewsID-Label" pairs (1=clicked, 0=not clicked) |

**Example Row:**
```
1	U87243	11/10/2019 11:30:54 AM	N8668 N39081 N65259	N78206-0 N94157-1 N78699-1
```

### 2. news.tsv

News article metadata including title, abstract, category, and extracted entities.

| Column | Name | Type | Description |
|--------|------|------|-------------|
| 1 | News ID | String | Unique article identifier (e.g., N1, N12345) |
| 2 | Category | String | Primary topic (e.g., sports, news, finance) |
| 3 | Subcategory | String | Secondary topic (e.g., football_nfl, markets) |
| 4 | Title | String | Article headline |
| 5 | Abstract | String | Article summary (may be empty) |
| 6 | URL | String | Original MSN link (links are expired) |
| 7 | Title Entities | JSON | Entities extracted from title |
| 8 | Abstract Entities | JSON | Entities extracted from abstract |

**Entity JSON Format:**
```json
[
  {
    "Label": "D. J. Reader",
    "Type": "P",
    "WikidataId": "Q24007178",
    "Confidence": 1.0,
    "OccurrenceOffsets": [24],
    "SurfaceForms": ["D.J. Reader"]
  }
]
```

Entity Types:
- `P` - Person
- `O` - Organization
- `G` - Geographic location
- `M` - Media/Publication
- `N` - Government/Institution
- `U` - Other/Unknown

### 3. entity_embedding.vec

Pre-trained entity embeddings from WikiData knowledge graph using TransE method.

| Column | Description |
|--------|-------------|
| 1 | WikiData entity ID (e.g., Q100, Q1000166) |
| 2-101 | 100-dimensional embedding vector (space-separated floats) |

**Example:**
```
Q100	-0.075855	-0.164252	0.128812	... (100 values total)
```

### 4. relation_embedding.vec

Pre-trained relation embeddings from WikiData knowledge graph.

| Column | Description |
|--------|-------------|
| 1 | Relation ID |
| 2-101 | 100-dimensional embedding vector |

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total impressions | 4,979,946 |
| Unique news articles | 130,379 |
| Entity embeddings | ~100,000 |
| Relation embeddings | ~100 |
| Collection period | Oct 12 - Nov 22, 2019 |
| Average history length | ~30 articles |
| Average impressions per session | ~20 articles |

## Parquet Conversion

Run `scripts/convert_to_parquet.py` to convert TSV files to Parquet format:

```bash
python data/mind_dataset/scripts/convert_to_parquet.py
```

This creates:
- `parquet/behaviors.parquet` - User impressions with parsed lists
- `parquet/news.parquet` - News metadata with parsed entity JSON
- `parquet/entity_embeddings.parquet` - Entity ID + embedding vector
- `parquet/relation_embeddings.parquet` - Relation ID + embedding vector

### Parquet Schema

**behaviors.parquet:**
```
impression_id: string
user_id: string
timestamp: string
history: list<string>
impression_news: list<string>
impression_labels: list<int64>
```

**news.parquet:**
```
news_id: string
category: string
subcategory: string
title: string
abstract: string
url: string
title_entities: list<struct>
abstract_entities: list<struct>
```

**entity_embeddings.parquet / relation_embeddings.parquet:**
```
id: string
embedding: list<float64>
```

## Validation

Run `scripts/validate_data.py` to verify parquet files:

```bash
python data/mind_dataset/scripts/validate_data.py
```

Validates:
- Parquet file structure and schema
- Data consistency (impression/label alignment)
- Label distribution and CTR
- Hash-based splitting functionality

## Key Differences from Alibaba PRM Dataset

| Aspect | MIND | Alibaba PRM |
|--------|------|-------------|
| Domain | News recommendation | E-commerce |
| Items per impression | Variable (avg ~20) | Fixed (30) |
| User history | Explicit click sequence | Not included |
| Feature semantics | Fully documented | Undocumented (proprietary) |
| Entity data | WikiData knowledge graph | None |
| Text content | Title + Abstract | None |

## License

The MIND dataset is released under the Microsoft Research License Terms. You must agree to these terms before using the dataset for research purposes.

## References

- **Original Paper:** Wu et al., "MIND: A Large-scale Dataset for News Recommendation", ACL 2020
- **Official Website:** https://msnews.github.io/
- **GitHub:** https://github.com/msnews/MIND
- **Azure Open Datasets:** https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news
