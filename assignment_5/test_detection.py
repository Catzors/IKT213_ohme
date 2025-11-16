# test_all.py
import face_recognition
import numpy as np
from PIL import Image
import sys

print("=" * 60)
print("COMPREHENSIVE FACE DETECTION TEST")
print("=" * 60)

# Test 1: Python version
print(f"\n1. Python version: {sys.version}")

# Test 2: Library versions
print(f"\n2. Library info:")
try:
    import dlib
    print(f"   dlib version: {dlib.__version__}")
except:
    print("   dlib version: unknown")

# Test 3: Test image loading
print(f"\n3. Testing image loading:")
test_images = [
    './known_people/Lars_v2.jpg',
    './known_people/Lars.jpg',
    './known_people/Lars_new_picture.jpg'
]

for img_path in test_images:
    try:
        img = face_recognition.load_image_file(img_path)
        print(f"   ✅ {img_path}")
        print(f"      Shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")
        
        # Test 4: HOG model
        print(f"      Testing HOG model...")
        faces_hog = face_recognition.face_locations(img, model="hog")
        print(f"      HOG found: {len(faces_hog)} face(s)")
        
        # Test 5: CNN model
        print(f"      Testing CNN model...")
        faces_cnn = face_recognition.face_locations(img, model="cnn")
        print(f"      CNN found: {len(faces_cnn)} face(s)")
        
        if len(faces_cnn) > 0:
            print(f"      Face locations: {faces_cnn}")
        
        print()
        
    except FileNotFoundError:
        print(f"   ❌ {img_path} - File not found")
    except Exception as e:
        print(f"   ❌ {img_path} - Error: {e}")

# Test 6: Test group photo
print(f"\n6. Testing group photo:")
try:
    image = face_recognition.load_image_file('./unknown_people/unkown_people.jpg')
    face_locations = face_recognition.face_locations(image)
    print(f"   Found {len(face_locations)} faces in group photo")
    for i, loc in enumerate(face_locations):
        print(f"   Face {i+1}: {loc}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
