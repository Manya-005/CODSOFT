# CODSOFT
import cv2
import face_recognition

# Load the pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load an example image with faces
image = cv2.imread('example.jpg')

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces using Haar cascades
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Initialize face recognition using dlib
face_encodings = face_recognition.face_encodings(image, faces)

# Load known face encodings (for recognition)
known_face_encodings = [your_known_face_encoding1, your_known_face_encoding2]
known_face_names = ["Person 1", "Person 2"]

# Iterate through detected faces
for (x, y, w, h), face_encoding in zip(faces, face_encodings):
    # Compare the detected face with known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # Find the known face with the smallest distance to the detected face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a rectangle and label on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the result
cv2.imshow('Face Detection and Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
