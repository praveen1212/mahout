package org.apache.mahout.clustering.streaming.tools;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.streaming.utils.ExperimentUtils;
import org.apache.mahout.clustering.streaming.utils.IOUtils;
import org.apache.mahout.utils.vectors.csv.CSVVectorIterable;

import java.io.IOException;

public class CSVToSequenceFile {
  private String inputFile;
  private String outputFile;

  public static void main(String[] args) {
    new CSVToSequenceFile().run(args);
  }

  public void run(String[] args) {
    if (!parseArgs(args)) {
      return;
    }

    Configuration conf = new Configuration();
    try {
      IOUtils.writeVectorsToSequenceFile(
          ExperimentUtils.castVectorsToCentroids(new CSVVectorIterable(true, new Path(inputFile), conf)),
          new Path(outputFile), conf);
    } catch (IOException e) {
      System.out.println(e.getMessage());
      e.printStackTrace();
    }
  }

  private boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFileOption = builder.withLongName("input")
        .withShortName("i")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
        .withDescription("where to get the CSV file")
        .create();

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("the output sequence file")
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(outputFileOption)
        .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 130));

    CommandLine cmdLine = parser.parseAndHelp(args);
    if (cmdLine == null) {
      return false;
    }

    inputFile = (String) cmdLine.getValue(inputFileOption);
    outputFile = (String) cmdLine.getValue(outputFileOption);
    return true;
  }
}
