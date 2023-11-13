package org.diplom.data;

import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.model.YOLO2;
import org.diplom.ImageUtils;
import org.diplom.yolo.Speed;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Stream;

public class TransformRotatingImages {
    private static final String BASE_PATH = "./src/main/resources/tripod-seq";
    private static final ImagePreProcessingScaler PRE_PROCESSING_SCALER = new ImagePreProcessingScaler(0, 1);
    private static final String TRANSFORMED = "./transformed";
    private static NativeImageLoader NATIVE_IMAGE_LOADER;

    public static void main(String[] args) throws IOException {
        takeOneForEachThree();
        cropAllWithYOLOBoundingBox();
        moveImageToTheLeft();
        resizeSomeOfImages();
        moveFilesToFolder();
        cleanUp();
    }

    private static void cleanUp() throws IOException {
        File file = new File(BASE_PATH + TRANSFORMED);
        String[] list = file.list();
        int count = 0;
        System.out.println("list.length = " + list.length);
        for (String fileName : list) {
            String pathname = BASE_PATH + TRANSFORMED + "\\" + fileName;
            String[] list1 = new File(pathname).list();
            if (list1 == null) {
                continue;
            }
            System.out.println(list1.length + " " + pathname);
            for (String s : list1) {
                File input = new File(pathname + "\\" + s);
                BufferedImage bufferedImage = ImageIO.read(input);
                if (bufferedImage == null) {
                    System.out.println("No Image " + input.getAbsolutePath());
                    input.delete();
                    continue;
                }
                if (bufferedImage.getWidth() <= 10 || bufferedImage.getHeight() <= 10) {
                    count++;
                    input.delete();
                }
            }
        }
        System.out.println("Cleaned  " + count);
    }

    private static void moveFilesToFolder() {
        File file = new File(BASE_PATH + TRANSFORMED);
        String[] list = file.list();
        for (String fileName : list) {
            String classFolder = fileName.substring(fileName.indexOf("q_") + 2, fileName.indexOf("q_") + 4);
            System.out.println("substring " + classFolder);
            File folder = new File(BASE_PATH + "\\" + TRANSFORMED + "\\" + classFolder);
            if (!folder.exists()) {
                folder.mkdirs();
            }
            new File(BASE_PATH + TRANSFORMED + "\\" + fileName)
                    .renameTo(new File(BASE_PATH + TRANSFORMED + "\\" + classFolder + "\\" + fileName));
        }
    }

    private static void resizeSomeOfImages() {
        File file = new File(BASE_PATH + TRANSFORMED);
        String[] list = file.list();
        List<String> imagesNames = Arrays.asList(list);
        Collections.shuffle(imagesNames);
        int size = (int) (imagesNames.size() * 0.3);
        List<String> strings = imagesNames.subList(0, size);
        strings.stream().parallel().forEach(fileName -> {
            String pathName = BASE_PATH + TRANSFORMED + "\\" + fileName;
            if (new File(pathName).isDirectory()) {
                return;
            }
            try {
                File input = new File(pathName);
                BufferedImage bufferedImage = ImageIO.read(input);
                double w = ThreadLocalRandom.current().nextInt(25, 40) / 100;
                double h = ThreadLocalRandom.current().nextInt(25, 40) / 100;

                BufferedImage scaledInstance = resizeImage(bufferedImage, (int) (bufferedImage.getWidth() * w), (int) (bufferedImage.getHeight() * h), bufferedImage.getType());
                File output = new File(BASE_PATH + TRANSFORMED + "\\" + fileName);
                ImageIO.write(scaledInstance, ".jpg", output);
                System.out.println("Resized " + output.getAbsolutePath());

            } catch (Exception e) {
                e.printStackTrace();
            }

        });
    }

    private static BufferedImage resizeImage(BufferedImage bufferedImage, int width, int height, int type) {
        BufferedImage resizedImage = new BufferedImage(width, height, type);
        Graphics2D graphics = resizedImage.createGraphics();
        graphics.drawImage(bufferedImage, 0, 0, width, height, null);
        graphics.dispose();
        return resizedImage;
    }

    private static void moveImageToTheLeft() throws IOException {
        File file = new File(BASE_PATH + TRANSFORMED);
        String[] list = file.list();
        for (String fileName : list) {
            String pathName = BASE_PATH + TRANSFORMED + "\\" + fileName;
            if (new File(pathName).isDirectory()) {
                continue;
            }
            BufferedImage bufferedImage = ImageIO.read(new File(pathName));
            int from = 0;
            for (int i = 0; i < 3; i++) {
                double percentage = ThreadLocalRandom.current().nextInt(20, 30) / 100;
                int x = (int) percentage * bufferedImage.getWidth();
                from = x + from;
                BufferedImage subImage = bufferedImage.getSubimage(from, 0, bufferedImage.getWidth() - from, bufferedImage.getHeight());
                ImageIO.write(subImage, "jpg", new File(BASE_PATH + TRANSFORMED + "\\" + fileName.replace(".jpg", "h_" + i + ".jpg")));
            }
        }
    }

    private static void cropAllWithYOLOBoundingBox() throws IOException {
        ComputationGraph yolo = (ComputationGraph) YOLO2.builder().build().initPretrained();
        File file = new File(BASE_PATH);
        File[] files = file.listFiles();
        sortFiles(files);
        Speed selectedSpeed = Speed.MEDIUM;
        NATIVE_IMAGE_LOADER = new NativeImageLoader(selectedSpeed.height, selectedSpeed.width, 3);
        Stream.of(files).parallel().forEach(e -> {
            try {
                cropImageWithYOLOBoundingBox(yolo, selectedSpeed, e);

            } catch (Exception e1) {
                e1.printStackTrace();
            }
        });
    }

    private static void cropImageWithYOLOBoundingBox(ComputationGraph yolo, Speed selectedSpeed, File file) throws IOException {
        if (file.isDirectory()) {
            return;
        }
        BufferedImage bufferedImage = ImageIO.read(file);
        INDArray features = NATIVE_IMAGE_LOADER.asMatrix(bufferedImage);
        Mat mat = NATIVE_IMAGE_LOADER.asMat(features);
        PRE_PROCESSING_SCALER.transform(features);
        INDArray result = yolo.outputSingle(features);
        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) yolo.getOutputLayer(0);
        List<DetectedObject> predictedObjects = outputLayer.getPredictedObjects(result, 0.5);
        YoloUtils.nms(predictedObjects, 0.5);
        Optional<DetectedObject> max = predictedObjects.stream().max((o1, o2) -> ((Double) o1.getConfidence()).compareTo(o2.getConfidence()));
        createCroppedImage(mat, selectedSpeed, max.get(), file);
    }

    private static void createCroppedImage(Mat mat, Speed selectedSpeed, DetectedObject obj, File file) throws IOException {
        double wPixelPerGrid = selectedSpeed.width / selectedSpeed.gridWidth;
        double hPixelPerGrid = selectedSpeed.height / selectedSpeed.gridHeight;

        double tx = Math.abs(obj.getTopLeftXY()[0] * wPixelPerGrid);
        double ty = Math.abs(obj.getTopLeftXY()[1] * hPixelPerGrid);
        BufferedImage image = ImageUtils.mat2BufferedImage(mat);

        double width = obj.getWidth();
        double height = obj.getHeight();

        if ((width * wPixelPerGrid + tx) > selectedSpeed.width) {
            width = (selectedSpeed.width + tx) / wPixelPerGrid;
        }

        if ((height * hPixelPerGrid + ty) > selectedSpeed.height) {
            height = (selectedSpeed.height + ty) / hPixelPerGrid;
        }

        File folder = new File(BASE_PATH + TRANSFORMED);

        if (!folder.exists()) {
            folder.mkdirs();
        }

        BufferedImage subImage = image.getSubimage((int) tx, (int) ty, (int) (wPixelPerGrid * width), (int) (hPixelPerGrid * height));
        ImageIO.write(subImage, "jpg", new File(BASE_PATH + "\\transformed\\" + file.getName()));
    }

    private static void takeOneForEachThree() {
        File file = new File(BASE_PATH);
        File[] files = file.listFiles();
        sortFiles(files);
        List<File> toBeDeleted = new ArrayList<>();
        int i = 0;
        String prevClassNumber = null;
        for (File file1 : files) {
            if (file1.isDirectory()) {
                continue;
            }

            String name = file1.getName();
            String number = name.substring(name.indexOf("seq_") + 4, name.indexOf(".jpg"));
            String classNumber = number.substring(0, number.indexOf("_"));
            if (i % 3 != 0) {
                toBeDeleted.add(file1);
            }
            if (prevClassNumber != null && !prevClassNumber.equals(classNumber)) {
                i = 0;
            }
            prevClassNumber = classNumber;
            i++;
        }
        toBeDeleted.stream().parallel().forEach(e -> e.delete());
    }

    private static void sortFiles(File[] files) {
        Arrays.sort(files, new Comparator<File>() {
            @Override
            public int compare(File o1, File o2) {
                if (o1.isDirectory() || o2.isDirectory()) {
                    return 1;
                }
                String number = extractNumberFromNAme(o1);
                String number2 = extractNumberFromNAme(o2);
                return Integer.valueOf(number).compareTo(Integer.valueOf(number2));
            }
        });
    }

    private static String extractNumberFromNAme(File o1) {
        //tripod_seq_01_002.jpg
        return o1.getName().substring(o1.getName().indexOf("seq_") + 4, o1.getName().indexOf(".jpg")).replace("_", "");
    }
}