package org.diplom.yolo;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.layers.convolutional.KerasSpaceToDepth;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.io.IOException;

public class YOLOPretrained {
    private static final long seed = 12345;
    private static final String TRAIN_PATH = "./src/main/resources/yolo2_dl4j_inference.v1.zip";
    private static final double[][] DEFAULT_PRIOR_BOXES = {{0.57273, 0.677385}, {1.87446, 2.06253}, {3.33843, 5.47434}, {7.88282, 3.52778}, {9.77052, 9.16828}};
    private static final double[][] priorBoxes = DEFAULT_PRIOR_BOXES;

    public static ComputationGraph initPretrained() throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        String filename = "./src/main/resources/yolo.h5";
        KerasLayer.registerCustomLayer("Lambda", KerasSpaceToDepth.class);
        ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(filename, false);
        INDArray priors = Nd4j.create(priorBoxes).castTo(DataType.FLOAT);
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(1e-3).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .build();

        System.out.println( "Configuration : " + fineTuneConf.toJson());

        ComputationGraph model = new TransferLearning.GraphBuilder(graph)
                .fineTuneConfiguration(fineTuneConf)
                .addLayer("outputs", new Yolo2OutputLayer.Builder()
                        .boundingBoxPriors(priors)
                        .build(), "conv2d_23")
                .setOutputs("outputs")
                .setInputTypes(InputType.convolutional(Speed.MEDIUM.height, Speed.MEDIUM.width, 3, CNN2DFormat.NCHW))
                .build();
        System.out.println(model.summary(InputType.convolutional(608, 608, 3)));
        ModelSerializer.writeModel(model, TRAIN_PATH, false);

        return ComputationGraph.load(new File(TRAIN_PATH), false);
    }
}
