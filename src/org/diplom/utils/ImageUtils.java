package org.diplom.utils;

import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.diplom.cifar.CifarImagePreProcessor;
import org.diplom.yolo.Speed;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

import static org.datavec.image.loader.CifarLoader.CHANNELS;

public class ImageUtils {
    public static final int HEIGHT = 32;
    public static  final int WIDTH = 32;

    public static final ParentPathLabelGenerator LABEL_GENERATOR_MAKER = new ParentPathLabelGenerator();

    public static BufferedImage mat2BufferedImage( Mat matrix) throws IOException {
        ByteBuffer allocate = ByteBuffer.allocate((int)matrix.arraySize());
        opencv_imgcodecs.imencode(".jpg", matrix, allocate);
        byte[] ba = allocate.array();
        BufferedImage bi = ImageIO.read(new ByteArrayInputStream(ba));
        return bi;
    }

    public static BufferedImage cropImageWithYOLO(Speed speed,
                                                  Mat mat,
                                                  DetectedObject obj,
                                                  boolean writeCropImageIntoDisk) throws IOException {
        double wPixelPerGrid = speed.width / speed.gridWidth;
        double hPixelPerGrid = speed.height/ speed.gridHeight;
        double tx = Math.abs(obj.getTopLeftXY()[0] * wPixelPerGrid);
        double ty = Math.abs(obj.getTopLeftXY()[1]*hPixelPerGrid);

        BufferedImage image = mat2BufferedImage(mat);

        double width = obj.getWidth();
        double height = obj.getHeight();

        if( (width * wPixelPerGrid) + tx > speed.width){
            width = (speed.width - tx)/wPixelPerGrid;
        }

        if (height*hPixelPerGrid + ty > speed.height){
            height = (speed.height - ty)/hPixelPerGrid;
        }

        BufferedImage subImage = image.getSubimage((int)tx, (int)ty, (int)(width*wPixelPerGrid), (int)(height*hPixelPerGrid));

        if (writeCropImageIntoDisk){
            ImageIO.write(subImage, "jpg",  new File("./src/main/resources/video_frames" + System.currentTimeMillis() + ".jpg"));
        }

        return subImage;
    }

    public static DataSetIterator createDataSetIterator(File sample, int numLabels, int batchSize) throws IOException {
        ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, LABEL_GENERATOR_MAKER );

        imageRecordReader.initialize(new FileSplit(sample));

        DataSetIterator iterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize, 1, numLabels);
        iterator.setPreProcessor(new CifarImagePreProcessor());
        return iterator;

    }

}
