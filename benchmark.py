# # File: benchmark.py
# import argparse
# import time
# import cv2
# import numpy as np

# from detector.yolo_detector import YOLODetector
# from detector.rcnn_detector import RCNNDetector
# from tracker.sort_tracker import Sort
# from speed.speed_estimator import SpeedEstimator
# from utils import Metrics

# def run_pipeline(detector, video_path):
#     cap = cv2.VideoCapture(video_path)
#     sort = Sort()
#     speed = SpeedEstimator()
#     metrics = Metrics()
#     counts = {'car': 0, 'motorcycle': 0, 'truck': 0}
#     speeders = 0

#     start = time.time()
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         boxes, _, labels = detector.detect(frame)
#         dets = [[int(x1),int(y1),int(x2),int(y2)]
#                 for x1,y1,x2,y2 in boxes]
#         dets_np = np.array(dets) if dets else np.empty((0,4))

#         tracks = sort.update(dets_np)
#         for x1, y1, x2, y2, tid in tracks:
#             cx, cy = (x1 + x2)//2, (y1 + y2)//2
#             spd = speed.update(tid, (cx, cy))
#             label = labels[0] if labels else 'car'
#             if label in counts and tid not in counts:
#                 counts[label] += 1
#             if spd > 30:
#                 speeders += 1

#         metrics.inc()
#     total_time = time.time() - start
#     cap.release()

#     fps = metrics.frames / total_time if total_time > 0 else 0
#     return {
#         'fps': fps,
#         'total_time': total_time,
#         'counts': counts,
#         'speeders': speeders
#     }

# def main():
#     p = argparse.ArgumentParser(
#         description="Benchmark YOLO-v8 vs Faster R-CNN on one video")
#     p.add_argument('--video', '-v', required=True,
#                    help="Path to .mp4 or .avi video")
#     p.add_argument('--true_counts', '-t', nargs=3, type=int, metavar=('CAR','MOTO','TRUCK'),
#                    help="Ground-truth counts: CAR MOTORCYCLE TRUCK")
#     args = p.parse_args()

#     for name, DetClass in [('YOLO-v8', YOLODetector), ('Faster R-CNN', RCNNDetector)]:
#         print(f"\n=== {name} ===")
#         det = DetClass()
#         res = run_pipeline(det, args.video)
#         print(f"FPS             : {res['fps']:.2f}")
#         print(f"Total time (s)  : {res['total_time']:.2f}")
#         print(f"Counts          : {res['counts']}")
#         print(f"Speeders (>30)  : {res['speeders']}")
#         if args.true_counts:
#             tc = dict(zip(['car','motorcycle','truck'], args.true_counts))
#             for cls in ['car','motorcycle','truck']:
#                 if tc[cls] > 0:
#                     acc = 100 * res['counts'].get(cls,0) / tc[cls]
#                     print(f"{cls.title():12} Accuracy: {acc:.1f}%")

# if __name__=='__main__':
#     main()


import argparse
import time
import cv2
import numpy as np

from detector.yolo_detector import YOLODetector
from detector.rcnn_detector import RCNNDetector
from tracker.sort_tracker import Sort
from speed.speed_estimator import SpeedEstimator
from utils import Metrics

def run_detector(detector, path):
    cap = cv2.VideoCapture(path)
    sort = Sort()
    speed = SpeedEstimator()
    metrics = Metrics()
    counts = {'car': 0, 'motorcycle': 0, 'truck': 0}
    speeders = 0

    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _, labels = detector.detect(frame)
        dets = [[int(x1),int(y1),int(x2),int(y2)] for x1,y1,x2,y2 in boxes]
        dets_np = np.array(dets) if dets else np.empty((0,4))

        tracks = sort.update(dets_np)
        for x1, y1, x2, y2, tid in tracks:
            cx, cy = (x1+x2)//2, (y1+y2)//2
            spd = speed.update(tid, (cx, cy))
            label = labels[0] if labels else 'car'
            if label in counts and tid not in counts:
                counts[label] += 1
            if spd > 30:
                speeders += 1

        metrics.inc()

    total_time = time.time() - start
    cap.release()

    fps = metrics.frames / total_time if total_time>0 else 0
    return {
        'fps': fps,
        'time': total_time,
        'counts': counts,
        'speeders': speeders
    }

def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLO-v8 vs Faster R-CNN on a video")
    parser.add_argument('-v','--video', required=True,
                        help="Path to a .mp4 or .avi video")
    parser.add_argument('-t','--true', nargs=3, type=int, metavar=('CAR','MOTO','TRUCK'),
                        help="True counts for CAR MOTORCYCLE TRUCK")
    args = parser.parse_args()

    results = {}
    for name, cls in [('YOLO-v8', YOLODetector), ('Faster R-CNN', RCNNDetector)]:
        print(f"\n--- Running with {name} ---")
        det = cls()
        res = run_detector(det, args.video)
        print(f"FPS             : {res['fps']:.2f}")
        print(f"Total Time (s)  : {res['time']:.2f}")
        print(f"Counts          : {res['counts']}")
        print(f"Speed >30 km/h  : {res['speeders']}")
        results[name] = res

        if args.true:
            tc = dict(zip(['car','motorcycle','truck'], args.true))
            for cls_name in tc:
                true_val = tc[cls_name]
                pred = res['counts'].get(cls_name,0)
                acc = 100 * pred / true_val if true_val>0 else 0
                print(f"{cls_name.title():12} Accuracy: {acc:.1f}%")

    # Final comparison summary
    print("\n=== SUMMARY COMPARISON ===")
    print(f"{'Model':12}  {'FPS':>6}  {'Time(s)':>8}  {'Car Acc(%)':>12}  {'Speeders':>9}")
    for name, res in results.items():
        car_acc = "-"
        if args.true:
            true_c = args.true[0]
            car_acc = f"{100*res['counts']['car']/true_c:.1f}" if true_c>0 else "-"
        print(f"{name:12}  {res['fps']:6.2f}  {res['time']:8.2f}  {car_acc:>12}  {res['speeders']:9}")

if __name__ == "__main__":
    main()
