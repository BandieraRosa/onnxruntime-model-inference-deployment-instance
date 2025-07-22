import cv2
import numpy as np
import onnxruntime as ort
import time

class Params:
    def __init__(self,
                 model_path,
                 img_size=(640, 640),
                 rect_conf_threshold=0.6,
                 iou_threshold=0.5,
                 classes=("Raw_Banana", "Raw_Mango", "Ripe_Banana", "Ripe_Mango"),
                 intra_op_num_threads=1,
                 log_severity_level=3):
        self.model_path = model_path
        self.img_size = img_size
        self.rect_conf_threshold = rect_conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = list(classes)
        self.intra_op_num_threads = intra_op_num_threads
        self.log_severity_level = log_severity_level

class Result:
    def __init__(self, class_id, confidence, box):
        self.class_id = class_id
        self.confidence = confidence
        self.box = box  # [x, y, w, h]

class Model:
    def __init__(self, params):
        self.params = params
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = params.intra_op_num_threads
        sess_opts.log_severity_level = params.log_severity_level
        self.session = ort.InferenceSession(params.model_path, sess_opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.WarmUpSession()

    def WarmUpSession(self):
        blank = np.zeros((self.params.img_size[0], self.params.img_size[1], 3), dtype=np.uint8)
        _ = self.RunSession(blank)

    def PreProcess(self, img: np.ndarray):
        if img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img_rgb.shape[:2]
        th, tw = self.params.img_size
        # Letterbox
        if w >= h:
            scale = w / tw
            new_w, new_h = tw, int(h / scale)
        else:
            scale = h / th
            new_h, new_w = th, int(w / scale)
        resized = cv2.resize(img_rgb, (new_w, new_h))
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]  # [1,3,H,W]
        return blob, scale

    def RunSession(self, img: np.ndarray):
        start_pre = time.time()
        blob, scale = self.PreProcess(img)
        pre_ms = (time.time() - start_pre) * 1000

        start_inf = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
        inf_ms = (time.time() - start_inf) * 1000

        start_post = time.time()
        results = self.PostProcess(outputs, scale)
        post_ms = (time.time() - start_post) * 1000

        print(f"[Model]: {pre_ms:.2f}ms pre-process, {inf_ms:.2f}ms inference, {post_ms:.2f}ms post-process")
        return results

    def PostProcess(self, output: np.ndarray, scale: float):
        # output shape: [1, dims, num_preds]
        pred = np.squeeze(output, axis=0)        # [dims, num_preds]
        pred = pred.T                            # [num_preds, dims]
        boxes, confidences, class_ids = [], [], []
        for row in pred:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf > self.params.rect_conf_threshold:
                x_c, y_c, w, h = row[:4]
                x = int((x_c - w/2) * scale)
                y = int((y_c - h/2) * scale)
                bw = int(w * scale)
                bh = int(h * scale)
                boxes.append([x, y, bw, bh])
                confidences.append(conf)
                class_ids.append(class_id)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.params.rect_conf_threshold, self.params.iou_threshold)
        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                results.append(Result(class_ids[i], confidences[i], boxes[i]))
        return results


def Detect(model: Model, img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    res = model.RunSession(img)
    for r in res:
        color = tuple(int(c) for c in np.random.randint(0, 256, 3))
        x, y, w, h = r.box
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
        label = f"{model.params.classes[r.class_id]} {r.confidence:.2f}"
        cv2.rectangle(img, (x, y-25), (x + len(label)*15, y), color, -1)
        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    cv2.imshow("Result of Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    params = Params(model_path="./model/best.onnx", rect_conf_threshold=0.1, iou_threshold=0.5)
    model = Model(params)
    Detect(model, "./images/Raw_Mango_0_5598.jpg")
