# Comparing embedding methods on famous Norwegian people

A single Colab notebook that extracts data on famous Norwegians from Wikidata and Norwegian Wikipedia, embeds their biographies with several methods, and compares the methods on classification, clustering, retrieval, and cost.

> **Status:** initial version. The notebook is written but results are not yet published here.

## What it does

1. **Extraction.** Name, birth date, birth place, and occupations come from Wikidata (SPARQL). The text to embed is the Norwegian Bokmål Wikipedia summary, with the Wikidata description as fallback.
2. **Labels.** Wikidata does not order occupations, so each person's occupations are ranked by how often they are mentioned in their Wikipedia article. The top 3 are kept and the top 1 is the main occupation. Main occupations with fewer than 30 people go to an `outros` class.
3. **Anti-leakage masking.** For classification and clustering, occupation terms are removed from the text (a lexicon mask over all occupation labels and aliases, plus a mask on the definition-sentence slot). Retrieval uses the unmasked text.
4. **Embeddings.**
   - TF-IDF (baseline)
   - Word2Vec trained on the corpus
   - Pretrained fastText (`cc.no.300`)
   - Multilingual sentence-transformers: `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-base`, `BAAI/bge-m3`
5. **Tasks and metrics.**

| Task | Setup | Metrics |
|---|---|---|
| Main-occupation classification | Stratified 5-fold CV; kNN (k=10) and logistic regression; majority-class baseline | Accuracy, macro-F1 (headline), weighted F1 |
| Clustering | KMeans, k = number of classes | NMI, ARI, silhouette |
| Semantic retrieval | 20 Norwegian queries, relevance derived automatically from Wikidata | Precision@5/10, MRR, nDCG@10 |
| Cost | Measured on the classification texts | Vector dimension, load and encode time, matrix and model size |

TF-IDF and Word2Vec are refit inside each CV fold on the training split only, so they get no unfair advantage over the pretrained models.

## How to run

1. Open `norwegian_people_embeddings.ipynb` in Google Colab.
2. Use a GPU runtime (*Runtime → Change runtime type → T4 GPU*) for the sentence-transformers.
3. Set your contact e-mail in `CONFIG["user_agent"]`. Wikimedia's User-Agent policy requires it.
4. Run all cells. If Colab asks to restart after the `pip install` cell, restart and continue from the next cell.

Raw downloads are cached in `cache/`, so re-running does not hit Wikidata or Wikipedia again. An interrupted Wikipedia download resumes where it stopped.

## Configuration

Everything adjustable lives in the `CONFIG` dict:

| Key | Meaning | Default |
|---|---|---|
| `min_sitelinks` | Threshold for "famous" (number of Wikipedia language editions) | 25 |
| `top_k_occupations` | Occupations kept per person | 3 |
| `min_class_size` | Smaller main-occupation classes become `outros` | 30 |
| `use_fasttext` | Download the pretrained vectors (about 7 GB, about 8 GB RAM). Set to `False` if memory runs out | `True` |
| `st_models` | Sentence-transformer models to compare | 3 models |
| `seed` | Random seed | 42 |

## Outputs

Written to `outputs/`:

| File | Content |
|---|---|
| `norwegians_data.csv` | People table (fields, top-3 occupations, label, original and masked text) |
| `metrics_extraction*.csv` | Extraction quality, top-15 occupations, births by century |
| `metrics_labels.csv` | Class distribution |
| `metrics_classification.csv` | Accuracy, macro-F1, weighted F1 (mean ± std) per method and classifier |
| `metrics_clustering.csv` | NMI, ARI, silhouette |
| `metrics_retrieval*.csv` | Retrieval metrics, per-query results, query diagnostics |
| `metrics_cost.csv` | Dimension, time, memory |
| `fig_comparison.png`, `fig_umap.png`, `fig_cost_quality.png` | Charts (labels in Portuguese) |
| `resumo.md` | Short summary in Portuguese, generated from the metrics |

The summary and chart labels are in Portuguese because the final report is written in Portuguese.

## Data sources and etiquette

- **Wikidata** (CC0) through the Wikidata Query Service. Queries are paginated, sequential, and retried with backoff, to stay under the 60 s query limit.
- **Norwegian Wikipedia** (CC BY-SA) through the MediaWiki API, using `wikipedia-api`.
- **fastText `cc.no.300`** (CC BY-SA 3.0).
- Requests carry a descriptive User-Agent and are rate-limited, following Wikimedia's API policies.

## Known limitations

- **Coverage bias.** The sitelinks threshold favors internationally visible people (athletes, politicians) and under-represents women and historical figures. Filtering on Norwegian citizenship can also miss people from the Denmark–Norway and Sweden–Norway union eras.
- **Heuristic labels.** The main occupation comes from mention counts, which can be wrong for people with parallel careers. The `outros` class is heterogeneous and mostly hurts clustering.
- **Masking is partial.** Explicit occupation terms are removed, but related words (for example *OL-gull* or *Stortinget*) stay. This is legitimate context, but it makes the task easier than fully blind classification.
- **Language.** Texts are mostly Bokmål, with some Nynorsk and short Wikidata descriptions. TF-IDF and Word2Vec use no lemmatization, and the multilingual models saw relatively little Norwegian in pretraining.
- **Automatic relevance.** Retrieval relevance comes from Wikidata, but a summary may not mention the attribute. See `mention_coverage_pct` in `metrics_retrieval_queries.csv`.
- **Corpus size.** With a few thousand short texts, Word2Vec trained from scratch is structurally disadvantaged against pretrained models.

## Repository layout

```
.
├── norwegian_people_embeddings.ipynb
├── README.md
├── cache/      # raw downloads (created on first run)
└── outputs/    # CSVs, figures, summary (created on first run)
```

## License

To be decided. Note that the data inherits the licenses of its sources (see above).
