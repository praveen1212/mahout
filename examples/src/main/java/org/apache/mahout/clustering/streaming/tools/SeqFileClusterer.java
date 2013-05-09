package org.apache.mahout.clustering.streaming.tools;

import java.io.IOException;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.clustering.ClusteringUtils;
import org.apache.mahout.clustering.streaming.cluster.StreamingKMeansThread;
import org.apache.mahout.clustering.streaming.mapreduce.StreamingKMeansDriver;
import org.apache.mahout.clustering.streaming.mapreduce.StreamingKMeansReducer;
import org.apache.mahout.clustering.streaming.utils.ExperimentUtils;
import org.apache.mahout.clustering.streaming.utils.IOUtils;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.neighborhood.ProjectionSearch;
import org.apache.mahout.math.stats.OnlineSummarizer;
import org.apache.mahout.utils.vectors.csv.CSVVectorIterable;

public class SeqFileClusterer {
  private String inputFile;
  private int numClusters;
  private DistanceMeasure distanceMeasure;
  private String outputFile;

  private Configuration conf = new Configuration();
  private boolean inCsv;

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

    Option inCsvOption = builder.withLongName("inCsv")
        .withShortName("c")
        .withArgument(argumentBuilder.withName("inCsv").withMaximum(1)
            .withDefault(false).create())
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(numClustersOption)
        .withOption(outputFileOption)
        .withOption(distanceOption)
        .withOption(inCsvOption)
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
    if (cmdLine.hasOption(inCsvOption)) {
      inCsv = true;
    }
    return true;
  }

  public static void main(String[] args) throws ClassNotFoundException {
    new SeqFileClusterer().run(args);
  }

  public List<Centroid> clusterKMeans(String inputFile) {
    List<Centroid> centroids = Lists.newArrayList();
    try {
      // Clean output.
      Path output = new Path("output");
      HadoopUtil.delete(conf, output);
      // Generate the random starting clusters.
      Path clusters = new Path("clusters");
      clusters = RandomSeedGenerator.buildRandom(conf, new Path(inputFile), clusters, numClusters, distanceMeasure);
      // Run KMeans.
      KMeansDriver.run(conf, new Path(inputFile), clusters, output, distanceMeasure, 0.01, numClusters, true, 0, true);
      // Read the results back in as a List<Centroid>.
      SequenceFileDirValueIterable<ClusterWritable> outIterable =
          new SequenceFileDirValueIterable<ClusterWritable>(
              new Path("output/clusters-*-final/part-*"), PathType.GLOB, conf);
      int numVectors = 0;
      for (ClusterWritable clusterWritable : outIterable) {
        centroids.add(new Centroid(numVectors++, clusterWritable.getValue().getCenter().clone(),
            clusterWritable.getValue().getNumObservations()));
      }
      System.out.printf("Clustered %d points\n", numVectors);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return centroids;
  }

  public List<Centroid> clusterKMeans(List<Centroid> datapoints) {
    List<Cluster> initialClusters = Lists.newArrayList();
    for (int i = 0; i < numClusters; ++i) {
      Centroid randomCentroid = datapoints.get(RandomUtils.nextInt(datapoints.size()));
      initialClusters.add(new Kluster(randomCentroid.getVector(), i, distanceMeasure));
    }
    ClusterClassifier prior = new ClusterClassifier(initialClusters,  new KMeansClusteringPolicy(0.01));
    ClusterClassifier clustered = ClusterIterator.iterate(Iterables.transform(datapoints, new Function<Centroid, Vector>() {
      @Override
      public Vector apply(Centroid input) {
        Preconditions.checkNotNull(input);
        return input.getVector();
      }
    }), prior, 10);

    List<Centroid> centroids = Lists.newArrayList();
    for (Cluster cluster : clustered.getModels()) {
      centroids.add(new Centroid(cluster.getId(), cluster.getCenter(), cluster.getTotalObservations()));
    }
    return centroids;
  }

  public List<Centroid> clusterBallStreamingKMeans(Iterable<Centroid> datapoints) {
    List<Centroid> finalCentroids = null;
    try {
      StreamingKMeansDriver.configureOptionsForWorkers(conf,
          numClusters,
          (numClusters * 20), (float) ClusteringUtils.estimateDistanceCutoff(datapoints, distanceMeasure, 100),
          20, 0.9f, false, false, 0.1f, 4,
          distanceMeasure.getClass().getName(), ProjectionSearch.class.getName(), 2, 3, true);

      // Run StreamingKMeans.
      System.out.printf("Starting StreamingKMeans.\n");
      List<Centroid> intermediateCentroids = Lists.newArrayList(new StreamingKMeansThread(datapoints, conf).call());

      // Run BallKMeans in final step.
      System.out.printf("Finished splitting centroids. Starting BallKMeans.\n");
      finalCentroids = Lists.newArrayList(ExperimentUtils.castVectorsToCentroids(
          StreamingKMeansReducer.getBestCentroids(intermediateCentroids, conf)));
    } catch (Exception e) {
      e.printStackTrace();
    }
    return finalCentroids;
  }

  public void printStatistics(String prefix, Iterable<Centroid> datapoints, List<Centroid> centroids) {
    List<OnlineSummarizer> summarizers =
        ClusteringUtils.summarizeClusterDistances(datapoints, centroids, distanceMeasure);
    System.out.printf("%s: Dunn Index %f\n", prefix, ClusteringUtils.dunnIndex(centroids, distanceMeasure, summarizers));
    System.out.printf("%s: Davies-Bouldin Index %f\n", prefix,
        ClusteringUtils.daviesBouldinIndex(centroids, distanceMeasure, summarizers));
  }

  public void printAdjustedRandIndex(Iterable<Centroid> datapoints,
                                     List<Centroid> rowCentroids, List<Centroid> columnCentroids) {
    Matrix confusionMatrix = ClusteringUtils.getConfusionMatrix(rowCentroids, columnCentroids,
        datapoints, distanceMeasure);
    for (int i = 0; i < confusionMatrix.numRows(); ++i) {
      for (int j = 0; j < confusionMatrix.numCols(); ++j) {
        System.out.printf("%3.0f ", confusionMatrix.get(i, j));
      }
      System.out.printf("\n");
    }
    System.out.printf("Adjusted Rand Index: %f\n", ClusteringUtils.getAdjustedRandIndex(confusionMatrix));
  }

  private void run(String[] args) {
    if (!parseArgs(args)) {
      System.out.printf("Unable to parse arguments\n");
      return;
    }

    try {
      conf.set("fs.default.name", "file:///");

      Iterable<Centroid> datapoints;
      if (inCsv) {
        datapoints = ExperimentUtils.castVectorsToCentroids(new CSVVectorIterable(false, new Path(inputFile), conf));
      } else {
        datapoints = ExperimentUtils.castVectorsToCentroids(
            IOUtils.getVectorsFromVectorWritableIterable(
                new SequenceFileValueIterable<VectorWritable>(new Path(inputFile), true, conf)));
      }

      List<Centroid> finalCentroidsKMeans;
      long start = System.currentTimeMillis();
      if (inCsv) {
        finalCentroidsKMeans = clusterKMeans(Lists.newArrayList(datapoints));
      } else {
        finalCentroidsKMeans = clusterKMeans(inputFile);
      }
      long end = System.currentTimeMillis();
      System.out.printf("Done. Took %f. Writing clusters to %s.\n", (end - start) / 1000.0, outputFile);


      IOUtils.writeVectorsToSequenceFile(finalCentroidsKMeans, new Path(outputFile + "-mahoutkmeans"), conf);

      List<Centroid> finalCentroidsBSKM;
      start = System.currentTimeMillis();
      finalCentroidsBSKM = clusterBallStreamingKMeans(datapoints);
      end = System.currentTimeMillis();
      System.out.printf("Done. Took %f. Writing clusters to %s.\n", (end - start) / 1000.0, outputFile);

      IOUtils.writeCentroidsToSequenceFile(finalCentroidsKMeans, new Path(outputFile), conf);

      printStatistics("mahout-kmeans", datapoints, finalCentroidsKMeans);
      printStatistics("ballstreaming-kmeans", datapoints, finalCentroidsBSKM);
      printAdjustedRandIndex(datapoints, finalCentroidsKMeans, finalCentroidsBSKM);
    } catch (IOException e) {
      System.out.printf("Unable to open file %s\n", inputFile);
      e.printStackTrace();
    } catch (Exception e) {
      System.out.printf("Something bad happened %s\n", e.getMessage());
      e.printStackTrace();
    }
  }
}
