package org.diplom.yolo;

import org.bytedeco.opencv.opencv_core.Mat;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.concurrent.atomic.AtomicInteger;

public class MarkedObject implements Comparable<MarkedObject> {

    private static final AtomicInteger ID = new AtomicInteger();
    private final long created;
    private final Mat frame;
    private DetectedObject detectedObject;
    private volatile INDArray l2Norm;
    private String id;
    private volatile boolean showed;

    public MarkedObject(DetectedObject detectedObject, INDArray l2Norm, long created, Mat frame) {
        this.detectedObject = detectedObject;
        this.l2Norm = l2Norm;
        this.created = created;
        this.frame = frame;

        id = "" + ID.incrementAndGet();

    }

    public long getCreated() {
        return created;
    }

    public Mat getFrame() {
        return frame;
    }

    public DetectedObject getDetectedObject() {
        return detectedObject;
    }

    public void setDetectedObject(DetectedObject detectedObject) {
        this.detectedObject = detectedObject;
    }

    public INDArray getL2Norm() {
        return l2Norm;
    }

    public void setL2Norm(INDArray l2Norm) {
        this.l2Norm = l2Norm;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public boolean isShowed() {
        return showed;
    }

    public void setShowed(boolean showed) {
        this.showed = showed;
    }

    @Override
    public String toString() {
        return "MarkedObject{" +
                "created=" + created +
                ", detectedObject=" + detectedObject +
                ", l2Norm=" + l2Norm +
                ", id='" + id + '\'' +
                '}';
    }

    @Override
    public int compareTo(MarkedObject o) {
        return Long.compare(o.created, o.created);
    }
}
