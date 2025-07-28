import pickle
import matplotlib.pyplot as plt

# Load the pickle file
file_path = "results/mnist_K5_stepsize0.1_stepsizeg0.1_stepsizep1_flag0_graphtypeerdos-renyi_n10.pickle"

with open(file_path, "rb") as f:
    results = pickle.load(f)

# First, show what type of object it is
print("Type of results:", type(results))

# If it's a list, let's look at elements
if isinstance(results, list):
    for i, item in enumerate(results):
        print(f"Item {i} type: {type(item)}")
        if isinstance(item, dict):
            print("Keys:", item.keys())

# Try plotting if it looks like a valid dict in the list
for item in results:
    if isinstance(item, dict):
        if "angle_dsa" in item:
            plt.plot(item["angle_dsa"], label="DSA")
        if "angle_fastpca" in item:
            plt.plot(item["angle_fastpca"], label="FAST-PCA")
        if "angle_seqdistpm" in item:
            plt.plot(item["angle_seqdistpm"], label="Sequential PM")

plt.xlabel("Iteration")
plt.ylabel("Projection Error / Cosine Angle Difference")
plt.title("Distributed PCA Methods Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
