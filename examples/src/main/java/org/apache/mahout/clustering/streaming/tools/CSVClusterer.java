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
import org.apache.mahout.clustering.streaming.cluster.StreamingKMeansThread;
import org.apache.mahout.clustering.streaming.mapreduce.StreamingKMeansDriver;
import org.apache.mahout.clustering.streaming.mapreduce.StreamingKMeansReducer;
import org.apache.mahout.clustering.streaming.utils.ExperimentUtils;
import org.apache.mahout.clustering.streaming.utils.IOUtils;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.neighborhood.ProjectionSearch;
import org.apache.mahout.utils.vectors.csv.CSVVectorIterable;

import java.io.IOException;
import java.util.List;

public class CSVClusterer {
  private String inputFile;
  private int numClusters;
  private DistanceMeasure distanceMeasure;
  private String outputFile;

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

    Option numClustersOption = builder.withLongName("numClusters")
        .withShortName("k")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("numClusters").withMaximum(1).create())
        .withDescription("the number of clusters to cluster the points in")
        .create();

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("where to dump the CSV file with the results")
        .create();

    Option distanceOption = builder.withLongName("distanceMeasure")
        .withShortName("dm")
        .withArgument(argumentBuilder.withName("distanceMeasure").withMaximum(1)
            .withDefault("org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure").create())
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(numClustersOption)
        .withOption(outputFileOption)
        .withOption(distanceOption)
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

    inputFile = (String) cmdLine.getValue(inputFileOption);
    numClusters = Integer.parseInt((String) cmdLine.getValue(numClustersOption));
    if (cmdLine.hasOption(outputFileOption)) {
      outputFile = (String) cmdLine.getValue(outputFileOption);
    }
    distanceMeasure = ClassUtils.instantiateAs(cmdLine.getValue(distanceOption).toString(), DistanceMeasure.class,
        new Class[]{}, new Object[]{});
    return true;
  }

  public static void main(String[] args) throws ClassNotFoundException {
    new CSVClusterer().run(args);
  }

  private void run(String[] args) {
    if (!parseArgs(args)) {
      System.out.printf("Unable to parse arguments\n");
      return;
    }

    try {
      Configuration conf = new Configuration();
      conf.set("fs.default.name", "file:///");
      CSVVectorIterable csvIterable = new CSVVectorIterable(true, new Path(inputFile), conf);

      // System.out.printf("%s\n", csvIterable.getHeader());
      Iterable<Centroid> datapoints = ExperimentUtils.castVectorsToCentroids(csvIterable);

      long start = System.currentTimeMillis();

      StreamingKMeansDriver.configureOptionsForWorkers(conf,
          numClusters,
          (numClusters * 20), 1e-5f,
          20, 0.9f, 10, distanceMeasure.getClass().getName(), ProjectionSearch.class.getName(), 2, 3, true);

      // Run StreamingKMeans.
      System.out.printf("Starting StreamingKMeans.\n");
      Iterable<Centroid> centroids = new StreamingKMeansThread(datapoints, conf).call();

      // Split intermediate centroids.
      System.out.printf("Finished StreamingKMeans. Splitting intermediate centroids into train and test.\n");
      Pair<List<Centroid>, List<Centroid>> split = StreamingKMeansReducer.splitTrainTest(centroids, conf);

      // Run BallKMeans in final step.
      System.out.printf("Finished splitting centroids. Starting BallKMeans.\n");
      centroids = ExperimentUtils.castVectorsToCentroids(
          StreamingKMeansReducer.getBestCentroids(split.getFirst(), split.getSecond(), conf));

      long end = System.currentTimeMillis();

      System.out.printf("Done. Took %f. Writing clusters to %s.\n", (end - start) / 1000.0, outputFile);
      IOUtils.writeCentroidsToSequenceFile(centroids, new Path(outputFile), conf);

    } catch (IOException e) {
      System.out.printf("Unable to open file %s\n", inputFile);
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      System.out.printf("Unable to instantiate class %s\n", e.getMessage());
      e.printStackTrace();
    } catch (Exception e) {
      System.out.printf("Something bad happened %s\n", e.getMessage());
      e.printStackTrace();
    }
  }
}
