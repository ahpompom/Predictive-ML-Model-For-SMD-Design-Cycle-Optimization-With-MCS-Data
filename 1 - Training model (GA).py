from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random

# Load the dataset
df = pd.read_csv('C:/Project 2/ComponentData_1 (1mil).csv')

# Convert boolean 'Core-Package conflict' into integer
df['Core-Package conflict'] = df['Core-Package conflict'].astype(int)

# Split data into features and target
X = df.drop('Core-Package conflict', axis=1)
y = df['Core-Package conflict']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# GA setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    # Apply feature selection based on the individual
    features = [i for i, bit in enumerate(individual) if bit == 1]
    if not features:
        return 0,  # Return a tuple
    
    X_train_selected = X_train[:, features]
    X_test_selected = X_test[:, features]
    
    # Train model
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_selected, y_train)
    
    # Evaluate model
    predictions = clf.predict(X_test_selected)
    return (accuracy_score(y_test, predictions),)

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Running the GA
population = toolbox.population(n=50)
NGEN = 40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

top_individual = tools.selBest(population, 1)[0]
best_features = [i for i, bit in enumerate(top_individual) if bit == 1]

# Train final model
X_train_selected = X_train[:, best_features]
X_test_selected = X_test[:, best_features]
final_clf = LogisticRegression(random_state=42)
final_clf.fit(X_train_selected, y_train)

# Predict and evaluate
predictions = final_clf.predict(X_test_selected)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')