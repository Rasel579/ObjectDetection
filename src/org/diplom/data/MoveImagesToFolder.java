package org.diplom.data;

import java.io.File;
import java.util.Optional;
import java.util.stream.Stream;

public class MoveImagesToFolder {
    private static final String PATHNAME = "\\VideoFolder";
    private static final int OFFSET = 405;

    public static void main(String[] args) {
        File[] files = new File(PATHNAME).listFiles();

        for (File file : files) {
            String name = file.getName();
            int i = name.indexOf("(");
            String folderNumber = null;
            if (i != -1) {
                folderNumber = name.substring(0, name.indexOf(" "));

            } else {
                continue;
            }
            String folder = String.valueOf(Integer.parseInt(folderNumber) + OFFSET);
            File newFile = new File(PATHNAME + "\\" + folder);
            if (newFile.exists()) {
                newFile.mkdirs();
            }
            file.renameTo(new File(PATHNAME + "\\" + folder + "\\" + file.getName()));
        }
        files = new File(PATHNAME).listFiles();

        for (File file : files) {
            if (file.isDirectory()) {
                String name = file.getName();
                Optional<File> first = Stream.of(files).parallel()
                        .filter(f -> f.getName().length() < 8 && !file.isDirectory() && removeJPG(f).equals("" + (Integer.parseInt(name) - OFFSET))).findFirst();
                if (!first.isPresent()) {
                    continue;
                }
                File file1 = first.get();

                file1.renameTo(new File(file.getAbsolutePath() + "\\" + file1.getName()));
            }
        }
    }

    private static String removeJPG(File file) {
        return file.getName().substring(0, file.getName().length() - 4);
    }
}
