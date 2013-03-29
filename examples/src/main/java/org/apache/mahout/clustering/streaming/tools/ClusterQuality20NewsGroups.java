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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.clustering.streaming.cluster.ClusteringUtils;
import org.apache.mahout.clustering.streaming.mapreduce.CentroidWritable;
import org.apache.mahout.clustering.streaming.mapreduce.StreamingKMeansDriver;
import org.apache.mahout.clustering.streaming.utils.ExperimentUtils;
import org.apache.mahout.clustering.streaming.utils.IOUtils;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.neighborhood.ProjectionSearch;
import org.apache.mahout.math.stats.OnlineSummarizer;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;

public class ClusterQuality20NewsGroups {
  private Configuration conf;
  private String inputFile;
  private String testInputFile;
  private String outputFile;
  private int projectionDimension = 0;

  private Pair<List<String>, List<Centroid>> input;
  private Pair<List<String>, List<Centroid>> testInput;
  private List<Centroid> reducedVectors;
  private List<Centroid> testReducedVectors;

  private int numRuns;

  private PrintWriter fileOut;

  private Path reducedInputPath = null;

  private int numPoints;
  private boolean clusterMahoutKMeans;
  private boolean clusterBallKMeans;
  private boolean clusterStreamingKMeans;
  private boolean mapReduce;

  private DistanceMeasure distanceMeasure;

  /**
   * Write the reduced vectors to a sequence file so that KMeans can read them.
   * @throws IOException
   */
  public void getReducedInputPath() throws IOException {
    reducedInputPath = new Path("hdfs://localhost:9000/tmp/input");
    HadoopUtil.delete(conf, reducedInputPath);
    SequenceFile.Writer seqWriter = SequenceFile.createWriter(FileSystem.get(conf), conf, reducedInputPath,
        Text.class, VectorWritable.class);
    for (int i = 0; i < reducedVectors.size(); ++i) {
      seqWriter.append(new Text(input.getFirst().get(i)), new VectorWritable(reducedVectors.get(i)));
    }
    seqWriter.close();
    System.out.printf("Wrote %d reduced vectors\n", reducedVectors.size());
  }

  public List<Centroid> clusterKMeans() {
    List<Centroid> kmCentroids = Lists.newArrayList();
    try {
      // If the reduced vectors haven't been written yet, write them.
      if (reducedInputPath == null) {
        getReducedInputPath();
      }
      // Clean output.
      Path output = new Path("hdfs://localhost:9000/tmp/output");
      HadoopUtil.delete(conf, output);
      // Generate the random starting clusters.
      Path clusters = new Path("hdfs://localhost:9000/tmp/clusters");
      clusters = RandomSeedGenerator.buildRandom(conf, reducedInputPath, clusters, 20, distanceMeasure);
      // Run KMeans.
      KMeansDriver.run(conf, reducedInputPath, clusters, output, distanceMeasure, 0.01, 20, true,
          0, true);
      // Read the results back in as a List<Centroid>.
      SequenceFileDirValueIterable<ClusterWritable> outIterable =
          new SequenceFileDirValueIterable<ClusterWritable>(
              new Path("hdfs://localhost:9000/tmp/output/clusters-*-final/part-*"), PathType.GLOB, conf);
      int numVectors = 0;
      for (ClusterWritable clusterWritable : outIterable) {
        kmCentroids.add(new Centroid(numVectors++, clusterWritable.getValue().getCenter().clone(),
            clusterWritable.getValue().getNumObservations()));
      }
      System.out.printf("Clustered %d points\n", numVectors);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return kmCentroids;
  }

  public void printSummariesInternal(List<OnlineSummarizer> summarizers, double time, String name, int numRun,
                                     String type) {
    System.out.printf("%s %s\n", name, type.toUpperCase());
    double maxDistance = 0;
    for (int i = 0; i < summarizers.size(); ++i) {
      OnlineSummarizer summarizer = summarizers.get(i);
      if (summarizer.getCount() == 0) {
        continue;
      }
      maxDistance = Math.max(maxDistance, summarizer.getMax());
      System.out.printf("Average distance in cluster %d [%d]: %f\n", i, summarizer.getCount(), summarizer.getMean());
      // If there is just one point in the cluster, quartiles cannot be estimated. We'll just assume all the quartiles
      // equal the only value.
      boolean moreThanOne = summarizer.getCount() > 1;
      if (fileOut != null) {
        fileOut.printf("%d,%s,%f,%d,%f,%f,%f,%f,%f,%f,%f,%d,%s\n", numRun, name, time, i, summarizer.getMean(),
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

  public void printSummaries(List<Centroid> centroids, double time, String name, int numRun) {
    printSummariesInternal(ClusteringUtils.summarizeClusterDistances(reducedVectors, centroids),
        time, name, numRun, "train");
    if (testReducedVectors != null) {
      printSummariesInternal(ClusteringUtils.summarizeClusterDistances(testReducedVectors, centroids),
          time, name, numRun, "test");
    }
  }

  public void runInstance(int numRun) throws ClassNotFoundException {
    System.out.printf("Run %d\n", numRun);
    long start;
    long end;
    double time;

    if (clusterMahoutKMeans) {
      System.out.printf("Clustering MahoutKMeans\n");
      start = System.currentTimeMillis();
      List<Centroid> kmCentroids = clusterKMeans();
      end = System.currentTimeMillis();
      time = (end - start) / 1000.0;
      System.out.printf("Took %f[s]\n", time);
      printSummaries(kmCentroids, time, "km", numRun);
    }

    if (clusterBallKMeans) {
      System.out.printf("Clustering BallKMeans k-means++\n");
      start = System.currentTimeMillis();
      List<Centroid> bkmCentroids =
          Lists.newArrayList(ExperimentUtils.clusterBallKMeans(reducedVectors, 20, 1, false, distanceMeasure));
      end = System.currentTimeMillis();
      time = (end - start) / 1000.0;
      System.out.printf("Took %f[s]\n", time);
      printSummaries(bkmCentroids, time, "bkm", numRun);

      System.out.printf("Clustering BallKMeans random centers\n");
      start = System.currentTimeMillis();
      List<Centroid> rBkmCentroids =
          Lists.newArrayList(ExperimentUtils.clusterBallKMeans(reducedVectors, 20, 1, true, distanceMeasure));
      end = System.currentTimeMillis();
      time = (end - start) / 1000.0;
      System.out.printf("Took %f[s]\n", time);
      printSummaries(rBkmCentroids, time, "bkmr", numRun);
    }

    if (clusterStreamingKMeans) {
      System.out.printf("Clustering StreamingKMeans with k [20] clusters\n");
      start = System.currentTimeMillis();
      List<Centroid> skmCentroids0 =
          Lists.newArrayList(ExperimentUtils.clusterStreamingKMeans(reducedVectors, 20, distanceMeasure));
      end = System.currentTimeMillis();
      time = (end - start) / 1000.0;
      System.out.printf("Took %f[s]\n", time);
      printSummaries(skmCentroids0, time, "skm0", numRun);

      int numStreamingClusters = (int) (20 * Math.log(reducedVectors.size()));
      System.out.printf("Clustering StreamingKMeans\n");
      start = System.currentTimeMillis();
      List<Centroid> skmCentroids =
          Lists.newArrayList(ExperimentUtils.clusterStreamingKMeans(reducedVectors, numStreamingClusters, distanceMeasure));
      end = System.currentTimeMillis();
      time = (end - start) / 1000.0;
      System.out.printf("Took %f[s]\n", time);
      printSummaries(skmCentroids, time, "skm", numRun);

      System.out.printf("Clustering OneByOneStreamingKMeans\n");
      start = System.currentTimeMillis();
      List<Centroid> oskmCentroids =
          Lists.newArrayList(ExperimentUtils.clusterOneByOneStreamingKMeans(reducedVectors, numStreamingClusters, distanceMeasure));
      end = System.currentTimeMillis();
      time = (end - start) / 1000.0;
      System.out.printf("Took %f[s]\n", time);
      printSummaries(oskmCentroids, time, "oskm", numRun);

      if (clusterBallKMeans) {
        System.out.printf("Clustering BallStreamingKMeans\n");
        start = System.currentTimeMillis();
        List<Centroid> bskmCentroids =
            Lists.newArrayList(ExperimentUtils.clusterBallKMeans(skmCentroids, 20, 0.9, true, distanceMeasure));
        end = System.currentTimeMillis();
        time += (end - start) / 1000.0;
        System.out.printf("Took %f[s]\n", time);
        printSummaries(bskmCentroids, time, "bskm", numRun);

        System.out.printf("Clustering OneByOneBallStreamingKMeans\n");
        start = System.currentTimeMillis();
        List<Centroid> boskmCentroids =
            Lists.newArrayList(ExperimentUtils.clusterBallKMeans(oskmCentroids, 20, 0.9, true, distanceMeasure));
        end = System.currentTimeMillis();
        time = (end - start) / 1000.0;
        System.out.printf("Took %f[s]\n", time);
        printSummaries(boskmCentroids, time, "boskm", numRun);
      }
    }

    if (mapReduce) {
      StreamingKMeansDriver.configureOptionsForWorkers(conf, 20,
          // StreamingKMeans
          200, 1e-10f,
          // BallKMeans
          20, 0.9f, 10,
          // Searcher
          CosineDistanceMeasure.class.getName(), ProjectionSearch.class.getName(), 10, 20);
      try {
        if (reducedInputPath == null) {
          getReducedInputPath();
        }
        Path output = new Path("streaming-centroids");
        HadoopUtil.delete(conf, output);
        start = System.currentTimeMillis();
        StreamingKMeansDriver.run(conf, reducedInputPath, output);
        end = System.currentTimeMillis();
        time = (end - start) / 1000.0;
        SequenceFileDirValueIterable<CentroidWritable> inputIterable =
            new SequenceFileDirValueIterable<CentroidWritable>(
                new Path("streaming-centroids/part-r*"), PathType.GLOB, conf);
        List<Centroid> mrCentroids = Lists.newArrayList();
        for (CentroidWritable centroidWritable : inputIterable) {
          mrCentroids.add(centroidWritable.getCentroid().clone());
        }
        printSummaries(mrCentroids, time, "mr", numRun);
      } catch (IOException e) {
        e.printStackTrace();
      } catch (InterruptedException e) {
        e.printStackTrace();
      } catch (ClassNotFoundException e) {
        e.printStackTrace();
      }
    }
  }

  public void run(String[] args) throws ClassNotFoundException {
    if (!parseArgs(args)) {
      return;
    }

    conf = new Configuration();
    conf.set("fs.default.name", "hdfs://localhost:9000/");
    try {
      Configuration.dumpConfiguration(conf, new OutputStreamWriter(System.out));

      System.out.printf("Reading training data\n");
      input = IOUtils.getKeysAndVectors(inputFile, projectionDimension, numPoints);
      reducedVectors = input.getSecond();
      System.out.printf("Read %d training vectors\n", reducedVectors.size());

      if (testInputFile != null) {
        System.out.printf("Reading test data\n");
        testInput = IOUtils.getKeysAndVectors(testInputFile, projectionDimension, numPoints);
        testReducedVectors = testInput.getSecond();
        System.out.printf("Read %d test vectors\n", testReducedVectors.size());
      }

      if (outputFile != null) {
        fileOut = new PrintWriter(new FileOutputStream(outputFile));
        fileOut.printf("run,type,time,cluster,distance.mean,distance.sd,distance.q0,distance.q1,distance.q2,distance.q3,"
          + "distance.q4,count,is.train\n");
      }
      for (int i = 0; i < numRuns; ++i) {
        runInstance(i);
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

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("where to dump the CSV file with the results")
        .create();

    Option projectOption = builder.withLongName("project")
        .withShortName("p")
        .withRequired(true)
        .withDescription("if set, projects the input vectors down to the requested number of dimensions")
        .withArgument(argumentBuilder.withName("project").withMaximum(1).create())
        .create();

    Option numPointsOption = builder.withLongName("numpoints")
        .withShortName("np")
        .withRequired(false)
        .withDescription("if set, only clusters the first numpoints points, otherwise it clusters all of them")
        .withArgument(argumentBuilder.withName("numpoints").withMaximum(1).create())
        .create();

    Option mahoutKMeansOption = builder.withLongName("mahoutKMeans")
        .withShortName("km")
        .withRequired(false)
        .withDescription("if set, clusters points using Mahout KMeans for comparison")
        .create();

    Option ballKMeansOption = builder.withLongName("ballKMeans")
        .withShortName("bkm")
        .withRequired(false)
        .withDescription("if set, clusters points using Ball KMeans for comparison")
        .create();

    Option streamingKMeansOption = builder.withLongName("streamingKMeans")
        .withShortName("skm")
        .withRequired(false)
        .withDescription("if set, clusters points using Streaming KMeans for comparison")
        .create();

    Option mapReduceOption = builder.withLongName("mapreduce")
        .withShortName("mr")
        .withRequired(false)
        .withDescription("if set, cluster points using Streaming KMeans + Ball KMeans as a MapReduce")
        .create();

    Option numRunsOption = builder.withLongName("numRuns")
        .withShortName("nr")
        .withRequired(false)
        .withArgument(argumentBuilder.withName("numRuns").withMaximum(1).withDefault(5).create())
        .create();

    Option distanceOption = builder.withLongName("distanceMeasure")
        .withShortName("dm")
        .withArgument(argumentBuilder.withName("distanceMeasure").withMaximum(1)
            .withDefault("org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure").create())
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(testInputFileOption)
        .withOption(outputFileOption)
        .withOption(projectOption)
        .withOption(numPointsOption)
        .withOption(mahoutKMeansOption)
        .withOption(ballKMeansOption)
        .withOption(streamingKMeansOption)
        .withOption(mapReduceOption)
        .withOption(numRunsOption)
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
    if (cmdLine.hasOption(testInputFileOption)) {
      testInputFile = (String) cmdLine.getValue(testInputFileOption);
    }
    if (cmdLine.hasOption(outputFileOption)) {
      outputFile = (String) cmdLine.getValue(outputFileOption);
    }
    if (cmdLine.hasOption(projectOption)) {
      projectionDimension = Integer.parseInt((String)cmdLine.getValue(projectOption));
    }
    if (cmdLine.hasOption(numPointsOption)) {
      numPoints = Integer.parseInt((String)cmdLine.getValue(numPointsOption));
    } else {
      numPoints = Integer.MAX_VALUE;
    }
    if (cmdLine.hasOption(mahoutKMeansOption)) {
      clusterMahoutKMeans = true;
    }
    if (cmdLine.hasOption(ballKMeansOption)) {
      clusterBallKMeans = true;
    }
    if (cmdLine.hasOption(streamingKMeansOption)) {
      clusterStreamingKMeans = true;
    }
    if (cmdLine.hasOption(mapReduceOption)) {
      mapReduce = true;
    }
    numRuns = Integer.parseInt(cmdLine.getValue(numRunsOption).toString());
    distanceMeasure = ClassUtils.instantiateAs(cmdLine.getValue(distanceOption).toString(), DistanceMeasure.class,
        new Class[]{}, new Object[]{});
    return true;
  }

  public static void main(String[] args) throws ClassNotFoundException {
    new ClusterQuality20NewsGroups().run(args);
  }
}
