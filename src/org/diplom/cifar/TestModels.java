package org.diplom.cifar;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.diplom.utils.ImageUtils;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Stream;

@Slf4j
public class TestModels {
    private static final String BASE = "./src/main/resources/models/";
    private static final String TEST_DATA = "./test_data";
    private static final double THRESHOLD = 0.85;
    private static final CifarImagePreProcessor IMAGE_PRE_PROCESSOR = new CifarImagePreProcessor();
    private static final NativeImageLoader LOADER = new NativeImageLoader(ImageUtils.HEIGHT, ImageUtils.WIDTH );
    private static boolean showTrainingPrecision = false;

    public static void main(String[] args) throws IOException {
        String[] allModels = new File(BASE).list();

        for (String model : allModels){
            ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(new File(BASE + model));
            String classesNumber = model.substring(0, model.indexOf("_"));
            log.info(vgg16.summary());
            if (showTrainingPrecision){
                showTrainingPrecision(vgg16, classesNumber);
            }
            TestResult testResult = test(vgg16, model);
            log.info(testResult.toString());
        }
    }

    private static void showTrainingPrecision(ComputationGraph vgg16, String classesNumber) throws IOException {
        File[] carTrackings = new File("CarTracking").listFiles();
        for (File carTracking : carTrackings){
           if (carTracking.getName().contains(classesNumber)){
               DataSetIterator dataSetIterator = ImageUtils.createDataSetIterator(carTracking, Integer.parseInt(classesNumber), 64);
               Evaluation eval = vgg16.evaluate(dataSetIterator);
               log.info(eval.stats());
           }
        }
    }

    public static TestResult test(ComputationGraph vgg16, String model){
        Map<File, List<INDArray>> map = buildEmbeddings(vgg16);
        int wrongPredictionsWithOtherClasses = 0;
        int wrongPredictionsInOneClassSequentially = 0;
        Set<Map.Entry<File, List<INDArray>>> entries = map.entrySet();

        for (Map.Entry<File, List<INDArray>> entry: entries) {
            wrongPredictionsWithOtherClasses += compareWithOtherClasses(entries, entry);
            wrongPredictionsInOneClassSequentially += compareInsideOneClassSequentially(entry);
        }

        return  new TestResult(
                model,
                wrongPredictionsInOneClassSequentially,
                wrongPredictionsWithOtherClasses,
                wrongPredictionsInOneClassSequentially + wrongPredictionsWithOtherClasses
        );
    }

    private static int compareWithOtherClasses(Set<Map.Entry<File, List<INDArray>>> entries, Map.Entry<File, List<INDArray>> entry) {
        int missMatch = 0;
        File folder = entry.getKey();
        List<INDArray> currentEmbeddings = entry.getValue();
        for (INDArray currentEmbedding: currentEmbeddings ){
            for (Map.Entry<File, List<INDArray>> entryOther : entries){
                if (entryOther.getKey().getName().equals(folder.getName())){
                    continue;
                }
                List<INDArray> otherEmbeddings = entryOther.getValue();
                for (INDArray otherEmbedding: otherEmbeddings){
                    if (currentEmbedding.distance2(otherEmbedding) < THRESHOLD){
                        missMatch++;
                    }
                }
            }
        }
        return missMatch;
    }

    private static int compareInsideOneClassSequentially(Map.Entry<File, List<INDArray>> entry){
        int wrongPredictionsInsideOneClassOrdered = 0;
        List<INDArray> value = entry.getValue();
        INDArray prevEmbedding = value.get(0);
        for (int i = 1; i < value.size(); i++){
            if (prevEmbedding.distance2(value.get(i)) > THRESHOLD){
                wrongPredictionsInsideOneClassOrdered++;
            }
            prevEmbedding = value.get(i);
        }
        return wrongPredictionsInsideOneClassOrdered;
    }

    private static Map<File, List<INDArray>> buildEmbeddings(ComputationGraph model) {
        File[] folders = new File(TEST_DATA).listFiles();
        Map<File, List<INDArray>> map = new HashMap<>();

        for (File folder : folders){
            File[] carIsOneClass = folder.listFiles();

            Arrays.sort(carIsOneClass, Comparator.comparing(File::getName));

            Map<File, INDArray> fileEmbedding = new TreeMap<>(Comparator.comparing(File::getName));

            Stream.of(carIsOneClass).forEach( inOneClass -> {
                try {
                    INDArray embeddings = getEmbeddings( model, inOneClass);
                    fileEmbedding.put(inOneClass, embeddings);
                }catch (Exception e){
                    e.printStackTrace();
                }
            });

            map.put(folder, new ArrayList<>(fileEmbedding.values()));
        }
        return map;
    }

    private static INDArray getEmbeddings(ComputationGraph model, File image) throws IOException {
        INDArray indArray = LOADER.asMatrix(image);
        IMAGE_PRE_PROCESSOR.preProcess(indArray);
        Map<String, INDArray> stringINDArrayMap = model.feedForward(indArray, false);
        INDArray embeddings = stringINDArrayMap.get("embeddings");
        return embeddings;
    }

    @AllArgsConstructor
    @Data
    static class TestResult {
        String model;
        int wrongPredictionInOneClassSequentially;
        int wrongPredictionsInOtherClasses;
        int total;
    };
}
