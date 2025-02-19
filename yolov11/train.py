import ultralytics


model = ultralytics.YOLO()

results = model.train(data='yolo-data.yml', epochs=12, imgsz=640, workers=0)