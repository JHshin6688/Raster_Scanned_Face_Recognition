# CV_project_face_recognition
In this project, I implemented an face classification model with the scikit-learn library, using PCA and LDA.

I utilized the face dataset in matrix format.

There are two data in face.mat.

- X
  - 520 face data saved in matrix format
  - size of X : 2576 * 520
  - By reshaping each (2576, 1) sized vector into (46, 56) sized vector, we can see the actual face image.

- l
  - labels of face data
  - size of l : 1 * 520
  - There are 52 types of faces, with 10 faces each


# Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique used to transform multidimensional data into a more compact form, extracting the main features of the data.

PCA analyzes the covariance structure of the data and transforms it into principal components.

# Linear discriminant analysis (LDA) 
LDA is another dimensionality reduction technique that finds a linear combination of features that characterizes or separates two or more classes of objects or events.

In LDA, betwee-class scatter matrix (𝑆𝑏) and within class scatter matrix (𝑆𝑤) are used to derive the feature space that best discriminates the train data based on their classes. This method is called Fisher Linear Discriminant (FLD).

The fisherface method consists of two phases.

First, we project the image set to a lower dimensional space, which is basically a PCA step. This ensures that that the resulting 𝑆𝑤 is nonsingular.

Then, we apply FLD to derive the optimal feature space. Using this feature space and k-neighbors classifier, we classified our test face images.
