import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import glob
# from basicsr.utils import imwrite
from mtcnn import MTCNN
from scipy.spatial.distance import cosine

# functions -------------------------------------------------------------------
class Facenet():
    def __init__(self, 
                 input='emergency\\test', 
                 output='facenet_results',
                 database='emergency/database',
                 facenet_model_path = 'models'):
        self.input = input
        self.output = output
        self.facenet_model_path = facenet_model_path

        self.database = database
        self.known_embeddings = None
        self.known_labels = None

        self.facenet_graph, self.input_tensor, self.output_tensor, self.phase_train_tensor = self.load_facenet_model(facenet_model_path)
        self.sess = tf.compat.v1.Session(graph=self.facenet_graph)
        os.sep = '/'
    
    # FACENET MODEL
    def load_facenet_model(self, model_path):
        try:
            facenet_graph = tf.Graph()
            with facenet_graph.as_default():
                graph_def = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(model_path, 'rb') as f:
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
            print("Facenet model loaded")
            input_tensor = facenet_graph.get_tensor_by_name('input:0')
            output_tensor = facenet_graph.get_tensor_by_name('embeddings:0')
            phase_train_tensor = facenet_graph.get_tensor_by_name('phase_train:0')
        except Exception as e:
            print(f"Error loading model: {e}")
            facenet_graph = None
            input_tensor = None
            output_tensor = None
            phase_train_tensor = None
        return facenet_graph, input_tensor, output_tensor, phase_train_tensor

    # FACE DETECTION MODEL
    def detect_faces_mtcnn(self, image, conf_threshold=0.9, mode="multiple"):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = MTCNN().detect_faces(rgb_image)
        
        faces = []
        confidences = []
        for result in results:
            confidence = result['confidence']
            x, y, width, height = result['box']
            startX, startY, endX, endY = x, y, x + width, y + height
            
            if confidence >= conf_threshold:
                face = image[startY:endY, startX:endX]
                faces.append((startX, startY, endX, endY, face))
                confidences.append(confidence)
        
        if mode=="multiple":
            return faces
        elif mode=="single":
            return faces[np.argmax(confidences)]

    # return a preprocessed face (returns only one)
    def preprocess_face(self, face, image_size=160):
        face = cv.resize(face, (image_size, image_size))
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        return face

    # return list of embeddings
    def generate_embeddings(self, faces):
        if self.facenet_graph is None:
            print("Facenet model is not loaded.")
            return None

        embeddings = []
        for face in faces:
            face = np.expand_dims(face, axis=0)
            feed_dict = {self.input_tensor: face, self.phase_train_tensor: False}
            embedding = self.sess.run(self.output_tensor, feed_dict=feed_dict)
            embedding = np.squeeze(embedding)  # Squeeze to ensure it's 1-D
            embeddings.append(embedding)
        
        print("Embedding complete.")
        return embeddings

    def cosine_similarity(embedding1, embedding2):
        return 1 - cosine(embedding1, embedding2)  # Similarity score (0 to 1)

    def recognize_faces(self, known_embeddings, known_labels, embeddings, threshold=0.4):
        recognized_faces = []
        for embedding in embeddings:
            distances = []
            for known_embedding in known_embeddings:
                distance = 1 - cosine(embedding, known_embedding)
                distances.append(distance)
            
            best_distance = np.max(distances)
            print(distances)
            if best_distance >= threshold:
                index = np.argmax(distances)
                recognized_faces.append((known_labels[index], best_distance))
            else:
                recognized_faces.append(("Unknown", best_distance))
        
        return recognized_faces
    
    def process_database(self):
        database_path = self.database
        self.known_labels = []
        known_images = []
        
        if database_path.endswith('/'):
            database_path = database_path[:-1]
        if os.path.isfile(database_path):
            database_list = [database_path]
        else:
            database_list = sorted(glob.glob(os.path.join(database_path, '*')))

        for db_path in database_list:
            db_path = db_path.replace("\\","/")
            print(f"Before method call, img_path: {db_path} (type: {type(db_path)})")
            img_name = os.path.basename(db_path)
            basename, ext = os.path.splitext(img_name)

            db_image = cv.imread(db_path)
            db_image = self.detect_faces_mtcnn(db_image, mode="single")
            known_images.append(db_image)
            self.known_labels.append(basename)

        known_faces = known_images
        preprocessed_known_faces = [self.preprocess_face(face) for _, _, _, _, face in known_faces]
        self.known_embeddings = self.generate_embeddings(preprocessed_known_faces)

    def run_recognition(self):
        self.process_database()
        
        if self.input.endswith('/') or self.input.endswith('\\'):
            self.input = self.input[:-1]
        if os.path.isfile(self.input):
            test_list = [self.input]
        else:
            test_list = sorted(glob.glob(os.path.join(self.input, '*')))

        for test_path in test_list:
            print(test_path)
            img_name = os.path.basename(test_path)
            basename, ext = os.path.splitext(img_name)

            image = cv.imread(test_path)
            faces = self.detect_faces_mtcnn(image)

            if len(faces) > 0:
                preprocessed_faces = [self.preprocess_face(face) for _, _, _, _, face in faces]
                embeddings = self.generate_embeddings(preprocessed_faces)

                recognized_faces = self.recognize_faces(self.known_embeddings, self.known_labels, embeddings)

                for (startX, startY, endX, endY, _), (label, distance) in zip(faces, recognized_faces):
                    cv.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 4)
                    cv.putText(image, f'{label} - {round(distance*100, 2)}%', (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
                
                cv.imwrite(f'{self.output}/{basename}_recognition{ext}', image)
            else:
                print("no faces found. repeating iteration.")
                cv.imwrite(f'{self.output}/{basename}_recognition{ext}', image)


# main -------------------------------------------------------------------
def main():
    FN = Facenet(input='set_input', 
                 output='set_output', 
                 database='database'
                )
    FN.run_recognition()

if __name__ == '__main__':
    main()

