import numpy as np

def polynomialFeatures( X, degree = 2, interaction_only = False, include_bias = True ) :
    features = X.copy()
    prev_chunk = X
    indices = list( range( len( X ) ) )

    for d in range( 1, degree ) :
        # Create a new chunk of features for the degree d:
        new_chunk = []
        # Multiply each component with the products from the previous lower degree:
        for i, v in enumerate( X[:-d] if interaction_only else X ) :
            # Store the index where to start multiplying with the current component
            # at the next degree up:
            next_index = len( new_chunk )
            for coef in prev_chunk[indices[i+( 1 if interaction_only else 0 )]:] :
                new_chunk.append( v*coef )
            indices[i] = next_index
        # Extend the feature vector with the new chunk of features from the degree d:
        features = np.append( features, new_chunk )
        prev_chunk = new_chunk

    if include_bias :
        features = np.insert( features, 0, 1 )

    return features

X = np.ones(shape=(1,4))
print(polynomialFeatures(X))
