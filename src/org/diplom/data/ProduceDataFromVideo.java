package org.diplom.data;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.diplom.ui.VideoPlayer;

import java.io.IOException;

public class ProduceDataFromVideo {
    public static void main(String[] args) throws IOException, InterruptedException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        VideoPlayer videoPlayer = new VideoPlayer();
        videoPlayer.startRealTimeVideoDetection(
                "./src/main/resources/videoSample.mp4",
                "",
                true,
                0.85,
                "631_epoch_data_e512_b256_600.zip",
                null
        );
    }
}
