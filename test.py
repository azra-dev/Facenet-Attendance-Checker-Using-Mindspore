# test.py
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from facenet_tensorflow import Facenet

def get_true_labels_and_predictions(facenet, test_dir):
    true_labels = []
    predicted_labels = []


    for identity_dir in sorted(glob.glob(os.path.join(test_dir, '*'))):
        if not os.path.isdir(identity_dir):
            continue  # Skip non-directory files


        true_label = os.path.basename(identity_dir)
        for image_path in sorted(glob.glob(os.path.join(identity_dir, '*'))):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            
            faces = facenet.detect_faces_mtcnn(image)
            
            if len(faces) > 0:
                preprocessed_faces = [facenet.preprocess_face(face) for _, _, _, _, face in faces]
                embeddings = facenet.generate_embeddings(preprocessed_faces)
                recognized_faces = facenet.recognize_faces(facenet.known_embeddings, facenet.known_labels, embeddings)
                
                # If a face is recognized, append the label, otherwise mark as "Unknown"
                if recognized_faces:
                    predicted_labels.append(recognized_faces[0][0])
                else:
                    predicted_labels.append("Unknown")
                
                # Append the true label for the image
                true_labels.append(true_label)
            else:
                print(f"No face detected in {image_path}.")
                predicted_labels.append("Unknown")
                true_labels.append(true_label)
    
    return true_labels, predicted_labels

def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    test_dir ='dataset/test'
    facenet = Facenet(
        input=test_dir, 
        output='facenet_test_results',
        database='dataset/test_db',
        facenet_model_path='models/20180402-114759.pb'
    )
    
    facenet.process_database()
    true_labels, predicted_labels = get_true_labels_and_predictions(facenet, test_dir)
    classes = facenet.known_labels + ["Unknown"]

    plot_confusion_matrix(true_labels, predicted_labels, classes)
