import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the pretrained FaceNet model
model_path = 'models/20180402-114759.pb'
training_path = 'dataset/train_eq'
validation_path = 'dataset/val_eq'
with tf.io.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

# Access the input and output tensors
input_tensor = graph.get_tensor_by_name('input:0')
output_tensor = graph.get_tensor_by_name('embeddings:0')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))
    img = img / 255.0  # Normalize to [0, 1]
    return img

def generate_embeddings(image_paths):
    embeddings = []
    with tf.compat.v1.Session(graph=graph) as sess:
        for image_path in image_paths:
            img = preprocess_image(image_path)
            img = np.expand_dims(img, axis=0) 
            
            feed_dict = {
                input_tensor: img,
            }
            
            # placeholder
            for op in graph.get_operations():
                if op.type == 'Placeholder' and 'phase_train' in op.name:
                    feed_dict[op.outputs[0]] = False
            
            emb = sess.run(output_tensor, feed_dict=feed_dict)
            embeddings.append(emb)
    return np.vstack(embeddings)


import os

def generate_labels(dataset_path):
    labels = []
    image_paths = []
    label_map = {}  # iter map
    current_label = 0

    for person_folder in os.listdir(dataset_path):
        person_folder_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_folder_path):
            if person_folder not in label_map:
                label_map[person_folder] = current_label
                current_label += 1
            
            for image_name in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_name)
                image_paths.append(image_path)
                labels.append(label_map[person_folder])
    
    return image_paths, labels, label_map


# Generate Labels
train_image_paths, train_labels, train_label_map = generate_labels(training_path)
val_image_paths, val_labels, val_label_map = generate_labels(validation_path)

# Generate Embedding
train_embeddings = generate_embeddings(train_image_paths)
val_embeddings = generate_embeddings(val_image_paths)

# Training
classifier = SVC(kernel='linear')
classifier.fit(train_embeddings, train_labels)

# Validating
val_predictions = classifier.predict(val_embeddings)
accuracy = accuracy_score(val_labels, val_predictions)
print(f'Validation accuracy: {accuracy}')


