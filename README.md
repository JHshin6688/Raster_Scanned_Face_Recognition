**In this project, I implemented a face classification model with PCA and LDA.**

## Data
I utilized the face dataset in matrix format, `face.mat`.
- `X`
  - 520 face images saved in matrix format
  - size of `X` : 2576 x 520 (each column is one flattened face image)
  - By reshaping each (2576, 1) sized column vector into a (46, 56) sized image, we can see the actual face image.

- `l`
  - labels of the face data
  - size of `l` : 1 x 520
  - There are 52 distinct people (classes), with 10 face images each

In the notebook, the data is loaded and split into a train/test set with an 8:2 ratio per class (8 images per person for training, 2 for testing).

## How to run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook PCA_LDA.ipynb
```

Run all cells in `PCA_LDA.ipynb` from top to bottom.  
Loads `face.mat`, builds the train/test split, runs a grid search over a list of `m_pca` / `m_lda` values, and re-runs the best combination with the full fisherface / recognition visualizations.  
Edit `m_pca_list` and `m_lda_list` to change which combinations are searched (`m_pca` must be <= `num_train_samples - num_classes`, and `m_lda` must be <= `num_classes - 1`).

## Functions in the Notebook
`show_gallery` : helper for displaying flattened face vectors as an image grid.  
`pca()` : computes the eigenfaces from the training data.  
`PCA_LDA()` : projects the data with PCA, then applies Fisher LDA to get the Fisherfaces.   
`PCA_LDA_classify()` : projects train/test data onto the Fisherface space and classifies with a k-nearest neighbor classifier.  
`PCA_LDA_grid_search()` : runs `PCA_LDA_classify()` for every `(m_pca, m_lda)` combination in the given lists, prints a table sorted by accuracy, and returns the best combination

**Result you will see:** the pipeline displays a gallery of Fisherfaces, the rows of `W_opt` reshaped into (46, 56) images — the discriminant directions that best separate the 52 identity classes.

## Face recognition results
`accuracy : 87.50% (91/104 correct)`
