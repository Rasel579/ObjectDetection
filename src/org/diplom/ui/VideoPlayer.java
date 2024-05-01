package org.diplom.ui;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.diplom.yolo.Strategy;
import org.diplom.yolo.Yolo;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.CountDownLatch;

import static org.bytedeco.opencv.global.opencv_highgui.destroyAllWindows;
import static org.bytedeco.opencv.global.opencv_highgui.waitKey;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

@Slf4j
public class VideoPlayer {
    private final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
    private Yolo yolo = new Yolo();
    private final CountDownLatch countDownLatch;
    private volatile boolean stop;

    public VideoPlayer() {
        countDownLatch = new CountDownLatch(2);
    }

    public void startRealTimeVideoDetection(
            String videoFileName,
            String windowName,
            boolean outputFrames,
            double threshold,
            String model,
            Strategy selectedItem
    ) throws Exception {
        log.info("Start video detection " + videoFileName);
        log.info(windowName);
        yolo.initialize(windowName, outputFrames, threshold, model, selectedItem);
        startYoloThread(yolo, windowName);
        countDownLatch.countDown();
        runVideoOnMainThread(yolo, windowName, videoFileName, converter);
    }

    private void runVideoOnMainThread(Yolo yolo, String windowName, String videoFileName, OpenCVFrameConverter.ToMat converter) throws FrameGrabber.Exception, InterruptedException {
        FFmpegFrameGrabber grabber = initGrabber(videoFileName);
        while (!stop) {
            Frame frame = grabber.grabFrame();
            if (frame == null) {
                log.info("stopping");
                stop();
                break;
            }
            if (frame.image == null) {
                continue;
            }
            Thread.sleep(60);
            Mat mat = converter.convert(frame);
            Mat resizeMat = new Mat(yolo.getSelectedSpeed().height, yolo.getSelectedSpeed().width, mat.type());
            yolo.push(resizeMat, windowName);
            resize(mat, resizeMat, resizeMat.size());
            yolo.drawBoundingBoxesRectangle(frame, resizeMat, windowName, null, null);
            char key = (char) waitKey(20);

            if (key == 27) {
                stop();
                break;
            }
        }
    }

    private FFmpegFrameGrabber initGrabber(String videoFileName) throws FFmpegFrameGrabber.Exception {
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(new File(videoFileName));
        grabber.start();
        return grabber;
    }

    private void startYoloThread(Yolo yolo, String windowName) {
        Thread thread = new Thread(() -> {
            while (!stop) {
                try {
                    yolo.predictBoundingBoxes(windowName);
                } catch (Exception e) {
                    log.error(e.getMessage());
                }
            }
            log.info("yolo thread exit");
        });

        thread.start();
    }

    public void stop() {
        if (!stop) {
            stop = true;
            yolo = new Yolo();
            destroyAllWindows();
        }
    }
}
