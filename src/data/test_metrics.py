
from shared.metrics import calculate_rmse, calculate_nasa_score

actual = [100, 80, 50, 20]
predicted = [95, 85, 45, 25]

print("RMSE:", calculate_rmse(actual, predicted))
print("NASA score:", calculate_nasa_score(actual, predicted))

# Perfect predictions should produce zero for both metrics.
assert calculate_rmse(actual, actual) == 0.0
assert calculate_nasa_score(actual, actual) == 0.0

print("Metric tests passed!")
