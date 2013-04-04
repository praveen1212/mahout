package org.apache.mahout.clustering.streaming.tools;

import com.google.common.collect.Lists;
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
import org.apache.mahout.clustering.streaming.cluster.RandomProjector;
import org.apache.mahout.clustering.streaming.utils.IOUtils;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.List;

public class SequenceFileToCSV {
  private String inputFile;
  private String outputFile;
  private String outputCSVFile;
  private Integer projectionDimension;

  public static void main(String[] args) {
    new SequenceFileToCSV().run(args);
  }

  public void run(String[] args) {
    if (!parseArgs(args)) {
      return;
    }

    Configuration conf = new Configuration();
    conf.set("fs.default.name", "hdfs://localhost:9000/");
    try {
      List<Centroid> vectors = Lists.newArrayList();
      Matrix projectionMatrix = null;
      int numVecs = 0;
      for (Vector vector : Lists.newArrayList(IOUtils.getVectorsFromVectorWritableIterable(
          new SequenceFileDirValueIterable<VectorWritable>(new Path(inputFile), PathType.LIST, conf)))) {
        if (projectionDimension != null) {
          if (projectionMatrix == null) {
            projectionMatrix = RandomProjector.generateBasisNormal(projectionDimension, vector.size());
          }
          vectors.add(new Centroid(numVecs++, projectionMatrix.times(vector), 1));
        } else {
          vectors.add(new Centroid(numVecs++, vector, 1));
        }
      }
      IOUtils.generateCSVFromVectors(vectors, conf, outputCSVFile);
      IOUtils.writeCentroidsToSequenceFile(vectors, conf, outputFile);
    } catch (IOException e) {
      System.out.println(e.getMessage());
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
        .withDescription("where to get seq files with the vectors")
        .create();

    Option outputCSVFileOption = builder.withLongName("outputcsv")
        .withShortName("oc")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("the output CSV file")
        .create();

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("the output sequence file")
        .create();

    Option projectOption = builder.withLongName("project")
        .withShortName("p")
        .withDescription("if set, projects the input vectors down to the requested number of dimensions")
        .withArgument(argumentBuilder.withName("project").withMaximum(1).create())
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(outputCSVFileOption)
        .withOption(outputFileOption)
        .withOption(projectOption)
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
    outputCSVFile = (String) cmdLine.getValue(outputCSVFileOption);
    if (cmdLine.hasOption(projectOption)) {
      projectionDimension = Integer.parseInt((String)cmdLine.getValue(projectOption));
    }
    return true;
  }
}
