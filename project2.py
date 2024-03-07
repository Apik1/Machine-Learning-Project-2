import numpy as np
import matplotlib.pyplot as plt

# Step 1, Create data
def generate_training_data():
    np.random.seed(0) 
    x = np.random.uniform(-1, 1, 20)  # Generate 20 random x values in the range [-1, 1]
    n = np.random.normal(0, 0.3, 20)  # Generate noise
    y = np.sin(2 * np.pi * x) + n  # Compute y values based on a sine function plus noise
    return x, y

# Assign data
x, y = generate_training_data()

# Prepare data for third order polynomial
X = np.vstack((np.ones(len(x)), x, x**2, x**3)).T

# Step 2, Linear Regression to learn parameters
def learning(X, y):
    X_transpose = X.T  
    estimated_params = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)      
    return estimated_params

# Assign and Print Learned Parameters
theta = learning(X, y)
print("Learned Parameters:", theta)

# Step 3, Make Predictions
x_prediction = np.array([0, 0.25, 0.5, 0.75, 1])
X_prediction = np.vstack((np.ones(len(x_prediction)), x_prediction, x_prediction**2, x_prediction**3)).T
y_prediction = X_prediction.dot(theta)
print("Predicted Output Values:", y_prediction)

# Step 4, Create Graph
# (1) Noiseless sine function
x_continuous = np.linspace(-1, 1, 400)
y_sine = np.sin(2 * np.pi * x_continuous)

# (4) Learned 3rd-order polynomial
y_poly = np.vstack((np.ones(len(x_continuous)), x_continuous, x_continuous**2, x_continuous**3)).T.dot(theta)

plt.figure(figsize=(10, 6))
plt.plot(x_continuous, y_sine, label='Noiseless Sine Function', color='green')
plt.scatter(x, y, label='Training Data', color='blue')
plt.scatter(x_prediction, y_prediction, label='Predicted Points', color='red')
plt.plot(x_continuous, y_poly, label='Learned 3rd-order Polynomial', color='orange')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Exploration')
plt.show()

### END P1
#######################################################################################################
### START P2

def cross_validate(X, y, k=5):
    np.random.seed(0)  
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    fold_size = int(len(X) / k)
    prediction_error_scores = []

    for fold in range(k):
        # Find start and end of Validation potion
        start, end = fold * fold_size, (fold + 1) * fold_size if fold < k-1 else len(X)
        
        # Split data into training or validation
        X_train = np.concatenate((X_shuffled[:start], X_shuffled[end:]), axis=0)
        y_train = np.concatenate((y_shuffled[:start], y_shuffled[end:]), axis=0)
        X_val = X_shuffled[start:end]
        y_val = y_shuffled[start:end]
        
        # Learn parameters
        theta_fold = learning(X_train, y_train)
        
        # Prediction based on validation set
        y_pred = X_val.dot(theta_fold)
        
        # Calculate root mean squated error for the fold
        prediction_error = np.sqrt(np.mean((y_val - y_pred)**2))
        prediction_error_scores.append(prediction_error)
        
        print(f"Fold {fold+1}: Learned Parameters: {theta_fold}, Prediction Error: {prediction_error}")
    
    # print the average rmse
    average_prediction_error = np.mean(prediction_error_scores)
    print("Average prediction error on all validation folds:", average_prediction_error)
    
    return prediction_error_scores, average_prediction_error

# Run func
prediction_error_scores, average_prediction_error = cross_validate(X, y)

### END P2
#########################################################
### Start P3

# Make a polynomial based on the order
def generate_polynomial(x, order):
    return np.vstack([x**i for i in range(order + 1)]).T

def cross_validate_with_order(x, y, order, k=5):
    # Perform 5 fold cross validation
    np.random.seed(0)  # Randomize nums
    indices = np.random.permutation(len(x))
    fold_size = len(x) // k
    errors = []

    for fold in range(k):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < k-1 else len(x)
        x_value, y_value = x[indices[start:end]], y[indices[start:end]]
        x_train, y_train = np.delete(x, indices[start:end]), np.delete(y, indices[start:end])

        X_train = generate_polynomial(x_train, order)
        X_value = generate_polynomial(x_value, order)

        theta = learning(X_train, y_train)
        y_pred = X_value.dot(theta)

        error = np.sqrt(np.mean((y_value - y_pred) ** 2))
        errors.append(error)

    return np.mean(errors)


x, y = generate_training_data()
orders = [1, 3, 5, 7, 9]
errors = []

# Find optimal order
for order in orders:
    error = cross_validate_with_order(x, y, order)
    errors.append(error)

optimal_order = orders[np.argmin(errors)]
print("Average Cross-Validation Error by Order:", errors)
print("Optimal Polynomial Order:", optimal_order)

# Plot
plt.figure(figsize=(10, 6))
for order in orders:
    X_poly = generate_polynomial(x, order)
    theta = learning(X_poly, y)
    x_continuous = np.linspace(-1, 1, 400)
    X_continuous_poly = generate_polynomial(x_continuous, order)
    y_poly = X_continuous_poly.dot(theta)
    plt.plot(x_continuous, y_poly, label=f'Order {order}')

plt.scatter(x, y, color='black', label='Training Data')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomials of Different Order')
plt.show()