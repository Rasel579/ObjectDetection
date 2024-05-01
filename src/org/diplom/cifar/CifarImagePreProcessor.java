package org.diplom.cifar;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

@Slf4j
public class CifarImagePreProcessor implements DataNormalization {

    private static final Nd4jBackend b = Nd4j.getBackend();

    public static final INDArray VGG_MEAN_OFFSET_BGR = Nd4j.create(new float[]{255, 255, 255});

    /**
     * Iterates over a dataset
     * accumulating statistics for normalization
     *
     * @param dataSetIterator the iterator to use for
     *                        collecting statistics.
     */
    @Override
    public void fit(DataSetIterator dataSetIterator) {

    }

    @Override
    public void preProcess(DataSet dataSet) {
        INDArray features = dataSet.getFeatures();
        this.preProcess(features);
    }

    public void preProcess(INDArray features) {
        Nd4j.getExecutioner().execAndReturn(
                new BroadcastAddOp(features.dup(), VGG_MEAN_OFFSET_BGR, features, 1)
        );
    }

    @Override
    public void transform(INDArray indArray) {
        this.preProcess(indArray);
    }

    @Override
    public void transform(INDArray indArray, INDArray indArray1) {
        this.transform(indArray);
    }

    @Override
    public void transformLabel(INDArray indArray) {

    }

    @Override
    public void transformLabel(INDArray indArray, INDArray indArray1) {
        transformLabel(indArray);
    }

    @Override
    public void revertFeatures(INDArray indArray) {
        Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(indArray.dup(), VGG_MEAN_OFFSET_BGR, indArray, 1));
    }

    @Override
    public void revertFeatures(INDArray indArray, INDArray indArray1) {
        revertFeatures(indArray);
    }

    @Override
    public void revertLabels(INDArray indArray) {

    }

    @Override
    public void revertLabels(INDArray indArray, INDArray indArray1) {
        revertLabels(indArray);
    }

    @Override
    public void fitLabel(boolean b) {
        if (b) {
            log.warn("Labels fitting not currently supported for ImagePreProcessingScaler. Labels will not be modified");
        }
    }

    @Override
    public boolean isFitLabel() {
        return false;
    }

    /**
     * Fit a dataset (only compute
     * based on the statistics from this dataset0
     *
     * @param dataSets the dataset to compute on
     */
    @Override
    public void fit(DataSet dataSets) {

    }

    @Override
    public void transform(DataSet dataSets) {

    }

    @Override
    public void revert(DataSet dataSets) {

    }

    @Override
    public NormalizerType getType() {
        return null;
    }
}
