from sklearn.metrics import accuracy_score  # For calculating accuracy
from utils import *  # Utility functions for data loading and model management
import system  # Module containing feature extraction functions


def main():
    # Load the trained model
    model = load_model()

    # Evaluate model on noisy test data
    noise_test_images, noise_test_labels = get_dataset("test")
    noise_test_feature_vectors = system.image_to_reduced_feature(noise_test_images)
    noise_test_predictions = model.predict(noise_test_feature_vectors)
    noise_test_accuracy = accuracy_score(noise_test_labels, noise_test_predictions)
    print(f"Accuracy on noise_test set: {noise_test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
