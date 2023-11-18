package org.diplom.yolo;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.diplom.ImageUtils;
import org.diplom.cifar.TrainCifar10Model;
import org.jetbrains.annotations.Nullable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.threadly.concurrent.collections.ConcurrentArrayList;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_imgproc.*;


@Slf4j
public class Yolo {
    private static final double YOLO_DETECTION_THRESHOLD = 0.7;
    public final String[] COCO_CLASSES = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    private final Map<String, Stack<Mat>> stackMap = new ConcurrentHashMap<>();
    private TrainCifar10Model trainCifar10Model = new TrainCifar10Model();
    private Speed selectedSpeed = Speed.SLOW;
    private boolean outputFrames;
    private double trackingThreshold;
    private String pretrainedCifarModel;
    private Strategy strategy;
    private volatile List<MarkedObject> predictedObjects = new ConcurrentArrayList<>();
    private volatile Set<MarkedObject> previousPredictedObjects = new TreeSet<>();

    private HashMap<Integer, String> map;
    private HashMap<String, String> groupMap;
    private Map<String, ComputationGraph> modelsMap = new ConcurrentHashMap<>();
    private NativeImageLoader loader;

    public void initialize(String windowName, boolean outputFrames, double threshold, String model, Strategy strategy) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        this.trackingThreshold = threshold;
        this.outputFrames = outputFrames;
        this.pretrainedCifarModel = model;
        this.strategy = strategy;
        stackMap.put(windowName, new Stack<>());

        ComputationGraph yolo;
        yolo = YOLOPretrained.initPretrained();

        prepareYOLOLabels();

        trainCifar10Model.loadTrainedModel(pretrainedCifarModel);
        modelsMap.put(windowName, yolo);
        loader = new NativeImageLoader(selectedSpeed.height, selectedSpeed.width, 3);
        warmUp(yolo);
    }

    private void warmUp(ComputationGraph model) throws IOException {
        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) model.getOutputLayer(0);
        BufferedImage read = ImageIO.read(new File("./src/main/resources/sample.jpg"));
        INDArray indArray = loader.asMatrix(read);
        indArray = prepareImage(indArray);
        INDArray results = model.outputSingle(indArray);
        outputLayer.getPredictedObjects(results, YOLO_DETECTION_THRESHOLD);
    }

    public void push(Mat frame, String windowName) {
        stackMap.get(windowName).push(frame);
    }

    public void drawBoundingBoxesRectangle(Frame frame, Mat matFrame, String windowName, CanvasFrame canvas, OpenCVFrameConverter.ToMat toMatConverter) {
        if (invalidData(frame, matFrame) || outputFrames) {
            return;
        }

        if (previousPredictedObjects == null) {
            previousPredictedObjects.addAll(predictedObjects);
        }

        ArrayList<MarkedObject> detectedObjects = predictedObjects == null ? new ArrayList<>() : new ArrayList<>(predictedObjects);

        for (MarkedObject markedObject : detectedObjects) {
            try {
                createBoundingBoxRectangle(matFrame, markedObject);
                if (toMatConverter != null && canvas != null) {
                    canvas.showImage(toMatConverter.convert(matFrame));
                } else {
                    imshow(windowName, matFrame);
                }
            } catch (Exception e) {
                e.printStackTrace();
                log.error("Problem with out of the bounds image");
                if (toMatConverter != null && canvas != null) {
                    canvas.showImage(toMatConverter.convert(matFrame));
                } else {
                    imshow(windowName, matFrame);
                }
            }
        }
        previousPredictedObjects.addAll(detectedObjects);
        if (detectedObjects.isEmpty()) {
            if (toMatConverter != null && canvas != null) {
                canvas.showImage(toMatConverter.convert(matFrame));
            } else {
                imshow(windowName, matFrame);
            }
        }

    }

    private void createBoundingBoxRectangle(Mat file, MarkedObject markedObject) {
        double[] xy1 = markedObject.getDetectedObject().getTopLeftXY();
        double[] xy2 = markedObject.getDetectedObject().getBottomRightXY();

        MarkedObject minDistanceMarkedObject = findExistanceCarMatch(markedObject);

        if (minDistanceMarkedObject == null) {
            minDistanceMarkedObject = markedObject;
        }
        String id = minDistanceMarkedObject.getId();
        int predictedClass = minDistanceMarkedObject.getDetectedObject().getPredictedClass();

        int w = selectedSpeed.width;
        int h = selectedSpeed.height;
        int x1 = (int) Math.round(w * xy1[0] / selectedSpeed.gridWidth);
        int y1 = (int) Math.round(h * xy1[1] / selectedSpeed.gridHeight);
        int x2 = (int) Math.round(w * xy2[0] / selectedSpeed.gridWidth);
        int y2 = (int) Math.round(h * xy2[1] / selectedSpeed.gridHeight);

        rectangle(file, new Point(x1, y1), new Point(x2, y2), org.bytedeco.opencv.opencv_core.Scalar.BLUE);
        putText(file, groupMap.get(map.get(predictedClass)) + "-" + id, new Point(x1 + 2, y1 - 2), FONT_HERSHEY_DUPLEX, 2, Scalar.GREEN);
    }

    @Nullable
    private MarkedObject findExistanceCarMatch(MarkedObject markedObject) {
        MarkedObject minDistanceMarkedObject = null;
        for (MarkedObject predictedObject : previousPredictedObjects) {
            double distance = predictedObject.getL2Norm().distance2(markedObject.getL2Norm());
            if (strategy == Strategy.IoU_PLUS_ENCODINGS) {
                if (YoloUtils.iou(markedObject.getDetectedObject(), predictedObject.getDetectedObject()) >= 0.5 && distance <= trackingThreshold) {
                    minDistanceMarkedObject = predictedObject;
                    markedObject.setId(minDistanceMarkedObject.getId());
                    break;
                } else if (strategy == Strategy.ONLY_ENCODINGS) {
                    if (distance < trackingThreshold) {
                        minDistanceMarkedObject = predictedObject;
                        markedObject.setId(minDistanceMarkedObject.getId());
                        break;
                    }
                } else if (strategy == Strategy.ONLY_IoU) {
                    if (YoloUtils.iou(markedObject.getDetectedObject(), predictedObject.getDetectedObject()) >= 0.4) {
                        minDistanceMarkedObject = predictedObject;
                        markedObject.setId(minDistanceMarkedObject.getId());
                        break;
                    }
                }
            }
        }
        return minDistanceMarkedObject;
    }

    private boolean invalidData(Frame frame, Mat matFrame) {
        return predictedObjects == null || matFrame == null || frame == null;
    }

    private INDArray prepareImage(Mat frame) throws IOException {
        if (frame == null) {
            return null;
        }
        ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
        INDArray indArray = loader.asMatrix(frame);
        if (indArray == null) {
            return null;
        }
        imagePreProcessingScaler.transform(indArray);
        return indArray;
    }

    private INDArray prepareImage(INDArray indArray) {
        if (indArray == null) {
            return null;
        }

        ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
        imagePreProcessingScaler.transform(indArray);
        return indArray;

    }

    private void prepareYOLOLabels() {
        prepareLabels(COCO_CLASSES);
    }

    private void prepareLabels(String[] coco_classes) {
        if (map == null) {
            groupMap = new HashMap<>();
            groupMap.put("car", "car");
            groupMap.put("bus", "bus");
            groupMap.put("truck", "truck");
            groupMap.put("person", "person");
            int i = 0;
            map = new HashMap<>();
            for (String s1 : coco_classes) {
                map.put(i++, s1);
            }
        }
    }

    public Speed getSelectedSpeed() {
        return selectedSpeed;
    }

    public void predictBoundingBoxes(String windowName) throws IOException {
        long start = System.currentTimeMillis();

        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) modelsMap.get(windowName).getOutputLayer(0);
        Mat matFrame = stackMap.get(windowName).pop();
        INDArray indArray = prepareImage(matFrame);
        log.info("stack of frame size " + stackMap.get(windowName).size());

        if (indArray == null) {
            return;
        }

        INDArray results = modelsMap.get(windowName).outputSingle(indArray);
        if (results == null) {
            return;
        }
        List<DetectedObject> predictedObjects = outputLayer.getPredictedObjects(results, YOLO_DETECTION_THRESHOLD);
        YoloUtils.nms(predictedObjects, 0.5);
        List<MarkedObject> markedObjects = predictedObjects.stream().filter(e -> groupMap.get(map.get(e.getPredictedClass())) != null).map(e -> {
            try {
                return new MarkedObject(e, trainCifar10Model.getEmbeddings(matFrame, e, selectedSpeed), System.currentTimeMillis(), matFrame);
            } catch (Exception e1) {
                e1.printStackTrace();
            }
            return null;
        }).collect(Collectors.toCollection(ConcurrentArrayList::new));

        if (outputFrames) {
            for (MarkedObject markedObject : markedObjects) {
                ImageUtils.cropImageWithYOLO(selectedSpeed, markedObject.getFrame(), markedObject.getDetectedObject(), outputFrames);
            }
        }
        this.predictedObjects = markedObjects;
        log.info("stack of predictions size " + this.predictedObjects.size());
        log.info("Prediction time " + (System.currentTimeMillis() - start) / 1000d);
    }

}
