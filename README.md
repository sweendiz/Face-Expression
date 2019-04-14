
    pip install tensorflow
    pip install opencv-python
	
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
	python retrain.py --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --architecture=MobileNet_1.0_224 --image_dir=images
      
    python label.py
