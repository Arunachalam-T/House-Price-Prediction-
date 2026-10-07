# Dataset files

`socal2.csv` is the observed CSV from the [Kaggle House Prices and Images - SoCal dataset](https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal). It has 15,474 rows and these eight columns:

`image_id`, `street`, `citi`, `n_citi`, `bed`, `bath`, `sqft`, `price`

The target is `price` in USD. The project uses `citi`, `bed`, `bath`, and `sqft` as standard tabular inputs. It does not use `image_id` (row/file key), `street` (high-cardinality free-text address), or `n_citi` (a label-encoded duplicate of the city). The supplied CSV contained no missing cells, but the pipeline still imputes missing predictor values.

The CSV is included so default training works without Kaggle authentication or a download. The image IDs were verified against all 15,474 archive filenames. A controlled HOG/color-feature comparison on the same held-out split performed worse than tabular-only, so images are not part of the delivered default model; the image-feature path remains available for experimentation. When `--use-images` is enabled and `data/socal_pics/` is absent, the project downloads the public Kaggle archive and extracts files named `<image_id>.jpg` into this folder. The archive is about 390 MB; the images are intentionally not bundled here. Alternatively, download the dataset from Kaggle and put the image files in `data/socal_pics/` yourself. The implementation verifies linkage by using the exact numeric `image_id` filename, never by guessing from row order.
