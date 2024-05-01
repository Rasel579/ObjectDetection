package org.diplom;

import org.diplom.ui.CarTrackingUI;
import org.diplom.ui.ProgressBar;

import javax.swing.*;
import java.util.concurrent.Executors;

public class Main {
    private static final JFrame mainFrame = new JFrame();
    public static void main(String[] args) {
        ProgressBar progressBar = new ProgressBar(mainFrame, true);
        progressBar.showProgressBar("Loading models");
        CarTrackingUI ui = new CarTrackingUI();
        Executors.newCachedThreadPool().submit(() -> {
           try {
               ui.initUi();
           } catch (Exception e){
               throw new RuntimeException(e);
           } finally {
               progressBar.setVisible(true);
               mainFrame.dispose();
           }
        });
    }
}