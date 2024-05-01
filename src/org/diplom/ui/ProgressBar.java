package org.diplom.ui;

import javax.swing.*;
import java.awt.*;

public class ProgressBar {
    private final JFrame mainFrame;
    private JProgressBar progressBar;
    private boolean unDecorated;

    public ProgressBar(JFrame mainFrame ){
        this.mainFrame = mainFrame;
        progressBar = createProgressBar(mainFrame);
    }

    public ProgressBar(JFrame mainFrame, boolean unDecorated){
        this.mainFrame = mainFrame;
        progressBar = createProgressBar(mainFrame);
        this.unDecorated = unDecorated;
    }

    public  void showProgressBar(String msg){
        SwingUtilities.invokeLater( () -> {
            if (unDecorated){
                mainFrame.setUndecorated(true);
            }
        });

        mainFrame.setLocationRelativeTo(null);

        progressBar = createProgressBar(mainFrame);
        progressBar.setString(msg);
        progressBar.setStringPainted(true);
        progressBar.setIndeterminate(true);
        progressBar.setVisible(true);

        mainFrame.add(progressBar, BorderLayout.NORTH);
        mainFrame.pack();
        mainFrame.setVisible(true);
        mainFrame.repaint();
    }

    private JProgressBar createProgressBar( JFrame mainFrame ){
        JProgressBar progressBar = new JProgressBar(JProgressBar.HORIZONTAL);
        progressBar.setVisible(false);
        mainFrame.add(progressBar, BorderLayout.NORTH);
        return  progressBar;
    }

    public void setVisible(boolean isVisible){
        progressBar.setVisible(isVisible);
    }
}
