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
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.streaming.cluster.ClusteringUtils;
import org.apache.mahout.clustering.streaming.mapreduce.CentroidWritable;
import org.apache.mahout.clustering.streaming.utils.IOUtils;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.stats.OnlineSummarizer;

import java.io.*;
import java.util.List;

public class SummarizeQuality20NewsGroups {
  private Configuration conf;
  private String outputFile;


  private PrintWriter fileOut;

  private String trainFile;
  private String testFile;
  private String centroidFile;
  private boolean mahoutKMeansFormat;

  public void printSummaries(List<OnlineSummarizer> summarizers, String type) {
    double maxDistance = 0;
    for (int i = 0; i < summarizers.size(); ++i) {
      OnlineSummarizer summarizer = summarizers.get(i);
      if (summarizer.getCount() == 0) {
        System.out.printf("Cluster %d is empty\n");
        continue;
      }
      maxDistance = Math.max(maxDistance, summarizer.getMax());
      System.out.printf("Average distance in cluster %d [%d]: %f\n", i, summarizer.getCount(), summarizer.getMean());
      // If there is just one point in the cluster, quartiles cannot be estimated. We'll just assume all the quartiles
      // equal the only value.
      boolean moreThanOne = summarizer.getCount() > 1;
      if (fileOut != null) {
        fileOut.printf("%d,%f,%f,%f,%f,%f,%f,%f,%d,%s\n", i, summarizer.getMean(),
            summarizer.getSD(),
            summarizer.getQuartile(0),
            moreThanOne ? summarizer.getQuartile(1) : summarizer.getQuartile(0),
            moreThanOne ? summarizer.getQuartile(2) : summarizer.getQuartile(0),
            moreThanOne ? summarizer.getQuartile(3) : summarizer.getQuartile(0),
            summarizer.getQuartile(4), summarizer.getCount(), type);
      }
    }
    System.out.printf("Num clusters: %d; maxDistance: %f\n", summarizers.size(), maxDistance);
  }

  public void run(String[] args) {
    if (!parseArgs(args)) {
      return;
    }

    conf = new Configuration();
    try {
      Configuration.dumpConfiguration(conf, new OutputStreamWriter(System.out));

      fileOut = new PrintWriter(new FileOutputStream(outputFile));
      fileOut.printf("cluster,distance.mean,distance.sd,distance.q0,distance.q1,distance.q2,distance.q3,"
          + "distance.q4,count,is.train\n");

      Iterable<Centroid> centroids;
      if (mahoutKMeansFormat) {
        SequenceFileDirValueIterable<ClusterWritable> clusterIterable =
            new SequenceFileDirValueIterable<ClusterWritable>(new Path(centroidFile), PathType.GLOB, conf);
        centroids = IOUtils.getCentroidsFromClusterWritableIterable(clusterIterable);
      } else {
        SequenceFileDirValueIterable<CentroidWritable> centroidIterable =
            new SequenceFileDirValueIterable<CentroidWritable>(new Path(centroidFile), PathType.GLOB, conf);
        centroids = IOUtils.getCentroidsFromCentroidWritableIterable(centroidIterable);
      }

      SequenceFileDirValueIterable<VectorWritable> trainIterable =
          new SequenceFileDirValueIterable<VectorWritable>(new Path(trainFile), PathType.GLOB, conf);
      printSummaries(ClusteringUtils.summarizeClusterDistances(
          IOUtils.getVectorsFromVectorWritableIterable(trainIterable), centroids), "train");

      if (testFile != null) {
        SequenceFileDirValueIterable<VectorWritable> testIterable =
            new SequenceFileDirValueIterable<VectorWritable>(new Path(testFile), PathType.GLOB, conf);
        printSummaries(ClusteringUtils.summarizeClusterDistances(
            IOUtils.getVectorsFromVectorWritableIterable(testIterable), centroids), "test");
      }

      if (outputFile != null) {
        fileOut.close();
      }
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
        .withDescription("where to get seq files with the vectors (training set)")
        .create();

    Option testInputFileOption = builder.withLongName("testInput")
        .withShortName("itest")
        .withArgument(argumentBuilder.withName("testInput").withMaximum(1).create())
        .withDescription("where to get seq files with the vectors (test set)")
        .create();

    Option centroidsFileOption = builder.withLongName("centroids")
        .withShortName("c")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("centroids").withMaximum(1).create())
        .withDescription("where to get seq files with the centroids (from Mahout KMeans or StreamingKMeansDriver)")
        .create();

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("where to dump the CSV file with the results")
        .create();

    Option mahoutKMeansFormatOption = builder.withLongName("mahoutkmeansformat")
        .withShortName("mkm")
        .withDescription("if set, read files as (IntWritable, ClusterWritable) pairs")
        .withArgument(argumentBuilder.withName("numpoints").withMaximum(1).create())
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(testInputFileOption)
        .withOption(outputFileOption)
        .withOption(centroidsFileOption)
        .withOption(mahoutKMeansFormatOption)
        .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 150));

    CommandLine cmdLine = parser.parseAndHelp(args);
    if (cmdLine == null) {
      return false;
    }

    trainFile = (String) cmdLine.getValue(inputFileOption);
    if (cmdLine.hasOption(testInputFileOption)) {
      testFile = (String) cmdLine.getValue(testInputFileOption);
    }
    centroidFile = (String) cmdLine.getValue(centroidsFileOption);
    outputFile = (String) cmdLine.getValue(outputFileOption);
    if (cmdLine.hasOption(mahoutKMeansFormatOption)) {
      mahoutKMeansFormat = true;
    }
    return true;
  }

  public static void main(String[] args) {
    new SummarizeQuality20NewsGroups().run(args);
  }
}
