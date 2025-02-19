import os
import ultralytics


P_WEIGHT = 'yolo11n.pt'
# P_WEIGHT = 'runs/detect/train/weights/best.pt'
P_TEST = 'datasets/demo-val/images'
P_RESULT = 'datasets/demo-val/results'

def main():

    targets = [f.split('.png')[0] for f in os.listdir(P_TEST)]

    if not os.path.exists(P_RESULT):
        os.makedirs(P_RESULT)

    model = ultralytics.YOLO(P_WEIGHT)

    # Run predictions on a list of images
    results = model([os.path.join(P_TEST, f + '.png') for f in targets])

    # Process results
    for i in range(len(results)):
        # results[i].show()  # Display the results
        results[i].save(os.path.join(P_RESULT, targets[i] + '.jpg'))  # Save the results


if __name__ == '__main__':
    main()
