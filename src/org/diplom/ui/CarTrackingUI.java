package org.diplom.ui;

import org.diplom.yolo.Strategy;

import javax.swing.*;
import javax.swing.plaf.FontUIResource;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

public class CarTrackingUI {
    public static final AtomicInteger atomicInteger = new AtomicInteger();
    private static final int FRAME_WIDTH = 550;
    private static final int FRAME_HEIGHT = 220;
    private static final Font FONT = new Font("Dialog", Font.BOLD, 18);
    private static final Font ITALIC = new Font("Dialog", Font.ITALIC, 18);
    private static final String AUTONOMOUS_DRIVING = "Car Tracking";
    private JFrame mainFrame;
    private JPanel mainPanel;
    private File selectedFile = new File("./src/main/resources/videoSample.mp4");
    private VideoPlayer videoPlayer;
    private ProgressBar progressBar;
    private JComboBox<String> chooserCifarModel;
    private JSpinner threshold;
    private JComboBox<Strategy> strategy;

    public void initUi() throws UnsupportedLookAndFeelException, ClassNotFoundException, InstantiationException, IllegalAccessException {
        adjustLookAndFeel();
        mainFrame = createMainFrame();
        mainPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));

        JPanel actionPanel = new JPanel();
        actionPanel.setBorder(BorderFactory.createTitledBorder(""));

        JButton chooseVideo = new JButton("Choose Video");
        chooseVideo.setBackground(Color.ORANGE.darker());
        chooseVideo.setForeground(Color.ORANGE.darker());
        chooseVideo.addActionListener(e -> chooseFileAction());

        actionPanel.add(chooseVideo);

        JButton start = new JButton("Start detection");

        start.setBackground(Color.GREEN.darker());
        start.setForeground(Color.GREEN.darker());
        start.addActionListener(e -> {
            progressBar = new ProgressBar(mainFrame);
            SwingUtilities.invokeLater(() -> progressBar.showProgressBar("Detecting video ..."));
            Executors.newSingleThreadExecutor().submit(() -> {
                try {
                    videoPlayer = new VideoPlayer();
                    Runnable runnable = () -> {
                        try {
                            videoPlayer.startRealTimeVideoDetection(
                                    selectedFile.getAbsolutePath(),
                                    String.valueOf(atomicInteger.incrementAndGet()),
                                    false,
                                    (Double) threshold.getValue(),
                                    (String) chooserCifarModel.getSelectedItem(),
                                    (Strategy) strategy.getSelectedItem()
                            );

                        } catch (Exception e2) {
                        }
                    };

                    new Thread(runnable).start();

                } catch (Exception e1) {
                    throw new RuntimeException(e1);
                } finally {
                    progressBar.setVisible(false);
                }
            });
        });

        actionPanel.add(start);

        JButton stop = new JButton("Stop");
        stop.setForeground(Color.BLUE.darker());
        stop.setBackground(Color.BLUE.darker());
        stop.addActionListener(e -> {
            if (videoPlayer == null) {
                return;
            }
            try {
                videoPlayer.stop();

            } catch (Exception e1) {
            }
            progressBar.setVisible(false);
        });
        actionPanel.add(stop);

        JButton cameraBtn = new JButton("Camera");
        actionPanel.add(cameraBtn);
        cameraBtn.addActionListener(e -> {
            progressBar = new ProgressBar(mainFrame);
            SwingUtilities.invokeLater(() -> progressBar.showProgressBar("Detecting camera ..."));
            Executors.newSingleThreadExecutor().submit(() -> {
                try {
                    WebCameraPlayer cameraPlayer = new WebCameraPlayer();
                    Runnable runnable = () -> {
                        try {
                            cameraPlayer.runCamera(
                                    String.valueOf(atomicInteger.incrementAndGet()),
                                    false,
                                    (Double) threshold.getValue(),
                                    (String) chooserCifarModel.getSelectedItem(),
                                    (Strategy) strategy.getSelectedItem()
                            );
                        } catch (Exception e2) {
                        }
                    };

                    new Thread(runnable).start();

                } catch (Exception e1) {
                    throw new RuntimeException(e1);
                } finally {
                    progressBar.setVisible(false);
                }
            });
        });
        mainPanel.add(actionPanel);

        chooserCifarModel = new JComboBox<>();
        chooserCifarModel.setForeground(Color.BLUE.darker());
        Stream.of(Objects.requireNonNull(new File("./src/main/resources/models").listFiles())).forEach(f1 -> chooserCifarModel.addItem(f1.getName()));
        JLabel label = new JLabel("Cifar-10-model");

        label.setForeground(Color.BLUE);
        mainPanel.add(label);
        mainPanel.add(chooserCifarModel);

        label = new JLabel("Threshold");
        label.setForeground(Color.DARK_GRAY);
        threshold = new JSpinner(new SpinnerNumberModel(0.9, 0.1, 2, 0.1));
        threshold.setFont(ITALIC);
        mainPanel.add(label);
        mainPanel.add(threshold);

        strategy = new JComboBox<>();
        Stream.of(Strategy.values()).forEach(e -> strategy.addItem(e));
        mainPanel.add(strategy);

        addSignature();

        mainFrame.add(mainPanel, BorderLayout.CENTER);
        mainFrame.setVisible(true);
    }

    private void chooseFileAction() {
        JFileChooser chooser = new JFileChooser();
        chooser.setCurrentDirectory(new File(new File("./src/main/resources").getAbsolutePath()));
        int action = chooser.showOpenDialog(null);
        if (action == JFileChooser.APPROVE_OPTION) {
            selectedFile = chooser.getSelectedFile();
        }
    }

    private void adjustLookAndFeel() throws UnsupportedLookAndFeelException, ClassNotFoundException, InstantiationException, IllegalAccessException {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        UIManager.put("Button.font", new FontUIResource(FONT));
        UIManager.put("Label.font", new FontUIResource(ITALIC));
        UIManager.put("Combobox.font", new FontUIResource(ITALIC));
        UIManager.put("ProgressBar.font", new FontUIResource(FONT));
    }

    private JFrame createMainFrame() {
        JFrame mainFrame = new JFrame();
        mainFrame.setTitle(AUTONOMOUS_DRIVING);
        mainFrame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        mainFrame.setSize(FRAME_WIDTH, FRAME_HEIGHT);
        mainFrame.setMaximumSize(new Dimension(FRAME_WIDTH, FRAME_HEIGHT));
        mainFrame.setLocationRelativeTo(null);
        mainFrame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent e) {
                System.exit(0);
            }
        });
        ImageIcon icon = new ImageIcon("icon.png");
        mainFrame.setIconImage(icon.getImage());
        return mainFrame;
    }

    private void addSignature() {
        JLabel signature = new JLabel("Р.М.Шайхисламов, студент гр. НТм(до)-22", SwingConstants.CENTER);
        signature.setFont(new Font(Font.SANS_SERIF, Font.ITALIC, 20));
        signature.setForeground(Color.BLUE);
        mainFrame.add(signature, BorderLayout.SOUTH);
    }

}
