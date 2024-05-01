package org.diplom.ui;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.IplImage;
import org.bytedeco.opencv.opencv_core.Mat;
import org.diplom.yolo.Strategy;
import org.diplom.yolo.Yolo;

import javax.swing.*;
import java.io.File;
import java.util.concurrent.CountDownLatch;

import static org.bytedeco.opencv.global.opencv_core.cvFlip;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.helper.opencv_imgcodecs.cvSaveImage;
@Slf4j
public class WebCameraPlayer extends JFrame {
   private CanvasFrame canvas = new CanvasFrame("Web Cam");
   private final OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
   private volatile boolean stop;
   private final CountDownLatch countDownLatch;
   private final int INTERVAL = 100;
   private Yolo yolo = new Yolo();

   public WebCameraPlayer(){
       canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
       countDownLatch = new CountDownLatch(2);
   }

   public void runCamera(String windowName,
                         boolean outputFrames,
                         Double threshold,
                         String model,
                         Strategy selectedItem) throws FrameGrabber.Exception {
       try(FrameGrabber grabber = new OpenCVFrameGrabber(0)) {

           yolo.initialize(windowName, outputFrames, threshold, model, selectedItem);
           startYoloThread(yolo, windowName);
           countDownLatch.countDown();

           OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
           IplImage img;
           int i = 0;
           new File("images").mkdir();
           grabber.start();

           while (true){
               Frame frame = grabber.grab();

               img = converter.convert(frame);
               cvFlip(img, img, 1);

               //cvSaveImage("images" + File.separator + (i++) + "-a.jpg", img);

               Thread.sleep(INTERVAL);

               Mat mat = toMatConverter.convert(frame);
               Mat resizeMat = new Mat(yolo.getSelectedSpeed().height, yolo.getSelectedSpeed().width, mat.type());
               yolo.push(resizeMat, windowName);
               org.bytedeco.opencv.global.opencv_imgproc.resize(mat, resizeMat, resizeMat.size());
               yolo.drawBoundingBoxesRectangle(frame, resizeMat, windowName, canvas, toMatConverter);
               //canvas.showImage(frame);
           }

       } catch ( Exception e){
           System.out.println(e.getMessage());
       }
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
}
