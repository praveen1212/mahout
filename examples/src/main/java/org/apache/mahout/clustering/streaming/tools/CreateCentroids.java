package org.apache.mahout.clustering.streaming.tools;

import com.google.common.base.Charsets;
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
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.streaming.cluster.StreamingKMeans;
import org.apache.mahout.clustering.streaming.utils.ExperimentUtils;
import org.apache.mahout.clustering.streaming.utils.IOUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.clustering.streaming.cluster.BallKMeans;
import org.apache.mahout.clustering.streaming.search.BruteSearch;
import org.apache.mahout.clustering.streaming.search.ProjectionSearch;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

public class CreateCentroids {
  private String inputFile;
  private String outputFileBase;
  private boolean computeActualCentroids;
  private boolean computeBallKMeansCentroids;
  private boolean computeStreamingKMeansCentroids;
  private Integer numClusters;

  public static void main(String[] args) throws IOException {
    CreateCentroids runner = new CreateCentroids();
    if (runner.parseArgs(args)) {
      runner.run(new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true));
    }
  }

  // TODO(dfilimon): Make more configurable.
  public static Pair<Integer, Iterable<Centroid>> clusterStreamingKMeans(
      Iterable<Centroid> dataPoints, int numClusters) {
    StreamingKMeans clusterer = new StreamingKMeans(new ProjectionSearch(new
        EuclideanDistanceMeasure(), 20, 10), numClusters, 10e-6);
    clusterer.cluster(dataPoints);
    return new Pair<Integer, Iterable<Centroid>>(clusterer.getCentroids().size(), clusterer);
  }

  // TODO(dfilimon): Make more configurable.
  public static Pair<Integer, Iterable<Centroid>> clusterBallKMeans(
      List<Centroid> dataPoints, int numClusters) {
    BallKMeans clusterer = new BallKMeans(new BruteSearch(new EuclideanDistanceMeasure()),
        numClusters, 20);
    clusterer.cluster(dataPoints);
    return new Pair<Integer, Iterable<Centroid>>(numClusters, clusterer);
  }

  public static Vector distancesFromCentroidsVector(Vector input, List<Centroid> centroids) {
    Vector encodedInput = new DenseVector(centroids.size());
    int i = 0;
    for (Centroid centroid : centroids) {
      encodedInput.setQuick(i, Math.exp(-input.getDistanceSquared(centroid)));
      System.out.printf("%f ", encodedInput.get(i));
      ++i;
    }
    System.out.printf("\n");
    return encodedInput;
  }

  private void run(PrintWriter printWriter) throws IOException {
    Configuration conf = new Configuration();
    SequenceFileDirIterable<Text, VectorWritable> inputIterable = new
        SequenceFileDirIterable<Text, VectorWritable>(new Path(inputFile), PathType.LIST, conf);

    if (computeActualCentroids) {
      printWriter.printf("Computing actual clusters\n");
      Map<String, Centroid> actualClusters = ExperimentUtils.computeActualClusters(inputIterable);
      String outputFile = outputFileBase + "-actual.seqfile";
      printWriter.printf("Writing actual clusters to %s\n", outputFile);
      IOUtils.writeCentroidsToSequenceFile(actualClusters.values(), conf, outputFile);
    }

    if (computeBallKMeansCentroids || computeStreamingKMeansCentroids) {
      List<Centroid> centroids =
          Lists.newArrayList(IOUtils.getCentroidsFromPairIterable(inputIterable));
      Pair<Integer, Iterable<Centroid>> computedClusterPair;
      String suffix;
      printWriter.printf("Computing clusters for %d points\n", centroids.size());
      if (computeBallKMeansCentroids) {
        computedClusterPair =  clusterBallKMeans(centroids, numClusters);
        suffix = "-ballkmeans";
      } else {
        computedClusterPair =  clusterStreamingKMeans(centroids, numClusters);
        suffix = "-streamingkmeans";
      }
      String outputFile = outputFileBase + suffix + ".seqfile";
      printWriter.printf("Writing %s computed clusters to %s\n", suffix, outputFile);
      IOUtils.writeCentroidsToSequenceFile(computedClusterPair.getSecond(), conf, outputFile);
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
        .withDescription("where to get test data (encoded with tf-idf)")
        .create();

    Option actualCentroidsOption = builder.withLongName("actual")
        .withShortName("a")
        .withDescription("if set, writes the actual cluster centroids to <output_base>-actual")
        .create();

    Option ballKMeansCentroidsOption = builder.withLongName("ballkmeans")
        .withShortName("bkm")
        .withDescription("if set, writes the ball k-means cluster centroids to " +
            "<output_base>-ballkmeans")
        .create();

    Option streamingKMeansCentroidsOption = builder.withLongName("streamingkmeans")
        .withShortName("skm")
        .withDescription("if set, writes the ball k-means cluster centroids to " +
            "<output_base>-streamingkmeans; note that the number of clusters for streaming " +
            "k-means is the estimated number of clusters and that no ball k-means step is " +
            "performed")
        .create();

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("the base of the centroids sequence file; will be appended with " +
            "-<algorithm> where algorithm is the method used to compute the centroids")
        .create();

    Option numClustersOption = builder.withLongName("numClusters")
        .withShortName("k")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("numClusters").withMaximum(1).create())
        .withDescription("the number of clusters to cluster the vectors in")
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(outputFileOption)
        .withOption(actualCentroidsOption)
        .withOption(ballKMeansCentroidsOption)
        .withOption(streamingKMeansCentroidsOption)
        .withOption(numClustersOption)
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
    outputFileBase = (String) cmdLine.getValue(outputFileOption);
    computeActualCentroids = cmdLine.hasOption(actualCentroidsOption);
    computeBallKMeansCentroids = cmdLine.hasOption(ballKMeansCentroidsOption);
    computeStreamingKMeansCentroids = cmdLine.hasOption(streamingKMeansCentroidsOption);
    numClusters = Integer.parseInt((String) cmdLine.getValue(numClustersOption));
    return true;
  }
}
