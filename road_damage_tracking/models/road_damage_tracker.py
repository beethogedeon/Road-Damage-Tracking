import torch
import numpy as np
import cv2
from road_damage_tracking.loaders import load_model
import supervision as sv
from pydantic import BaseModel


class Response(BaseModel):
    frame: np.ndarray
    diagram: dict


class RoadDamageDetector:

    def __init__(self, source, height=640, width=640):

        self.labels = []
        self.source = source
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
        self.diagram = {"0": [],
                        "1": [],
                        "2": [],
                        "3": [],
                        "4": [],
                        "5": [],
                        }
        self.height = height
        self.width = width
        self.checkline = [(0, self.height * 0.75), (self.width, self.height * 0.75)]

    def predict(self, frame):
        results = self.model.track(frame)

        return results

    def check(self, detections: sv.Detections):

        for i in range(6):
            if i in detections.class_id:
                for detection in detections:
                    xyxy, mask, confidence, class_id, tracker_id = detection

                    # y_bottom_center = (xyxy[1] + xyxy[3]) / 2

                    if xyxy[3] == 0.75:

                        self.diagram[str(i)].append(True)

                    else:
                        self.diagram[str(i)].append(False)
            else:
                self.diagram[str(i)].append(False)

    def plot_bboxes(self, results, frame):

        cv2.line(frame, self.checkline[0], self.checkline[1], (46, 162, 112), 3)
        # xyxys = []
        # confidences = []
        # class_ids = []

        # for result in results:
        #    xyxys.append(result.boxes.xyxy.cpu().numpy())
        #    confidences.append(result.boxes.conf.cpu().numpy())
        #    class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        # Setup detections for visualization
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for _, confidence, class_id, tracker_id
                       in detections]

        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        self.check(detections)

        yield frame, self.diagram

    def __call__(self):

        cap = cv2.VideoCapture(self.source)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        timer = 0


        while True:

            # start_time = time()

            ret, frame = cap.read()
            frame.__index__()
            assert ret

            results = self.predict(frame)
            frame, diagram = self.plot_bboxes(results, frame)

            # end_time = time()
            # fps = 1 / np.round(end_time - start_time, 2)

            # cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # cv2.imshow('Road Damage Detection', frame)

            response = Response(frame=frame,diagram=diagram)

            yield response

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
