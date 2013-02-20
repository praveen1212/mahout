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
import org.apache.mahout.clustering.streaming.utils.ExperimentUtils;
import org.apache.mahout.clustering.streaming.utils.IOUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.stats.OnlineSummarizer;

import java.io.*;
import java.util.List;

public class ClusterQuality20NewsGroups {
  private final static int NUM_RUNS = 6;

  private Configuration conf;
  private String inputFile;
  private String outputFile;
  private int projectionDimension;
  private Pair<List<String>, List<Centroid>> input;
  private List<Centroid> reducedVectors;

  private PrintWriter fileOut;

  private Path reducedInputPath = null;

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
      clusters = RandomSeedGenerator.buildRandom(conf, reducedInputPath, clusters, 20, new EuclideanDistanceMeasure());
      // Run KMeans.
      KMeansDriver.run(conf, reducedInputPath, clusters, output, new EuclideanDistanceMeasure(), 0.01, 20, true,
          0, true);
      // Read the results back in as a List<Centroid>.
      SequenceFileDirValueIterable<ClusterWritable> outIterable =
          new SequenceFileDirValueIterable<ClusterWritable>(
              new Path("hdfs://localhost:9000/tmp/output/clusters-20-final/part-*"), PathType.GLOB, conf);
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

  public void printSummaries(List<OnlineSummarizer> summarizers, double time, String name, int numRun) {
    for (int i = 0; i < summarizers.size(); ++i) {
      OnlineSummarizer summarizer = summarizers.get(i);
      System.out.printf("Average distance in cluster %d [%d]: %f\n", i, summarizer.getCount(), summarizer.getMean());
      fileOut.printf("%d, %s, %f, %d, %f, %f, %d\n", numRun, name, time, i, summarizer.getMean(),
          summarizer.getSD(), summarizer.getCount());
    }
  }

  public void runInstance(int numRun) {
    System.out.printf("Run %d\n", numRun);

    System.out.printf("Clustering MahoutKMeans\n");
    long start = System.currentTimeMillis();
    List<Centroid> kmCentroids = clusterKMeans();
    long end = System.currentTimeMillis();
    double time = (end - start) / 1000.0;
    System.out.printf("Took %f[s]\n", time);
    List<OnlineSummarizer> summarizers = ExperimentUtils.summarizeClusterDistances(reducedVectors, kmCentroids);
    printSummaries(summarizers, time, "km", numRun);

    System.out.printf("Clustering BallKMeans\n");
    start = System.currentTimeMillis();
    List<Centroid> bkmCentroids = Lists.newArrayList(ExperimentUtils.clusterBallKMeans(reducedVectors, 20));
    end = System.currentTimeMillis();
    time = (end - start) / 1000.0;
    System.out.printf("Took %f[s]\n", time);
    summarizers = ExperimentUtils.summarizeClusterDistances(reducedVectors, bkmCentroids);
    printSummaries(summarizers, time, "bkm", numRun);

    System.out.printf("Clustering StreamingKMeans\n");
    start = System.currentTimeMillis();
    List<Centroid> skmCentroids = Lists.newArrayList(ExperimentUtils.clusterStreamingKMeans(reducedVectors, 20));
    end = System.currentTimeMillis();
    time = (end - start) / 1000.0;
    System.out.printf("Took %f[s]\n", time);
    summarizers = ExperimentUtils.summarizeClusterDistances(reducedVectors, skmCentroids);
    printSummaries(summarizers, time, "skm", numRun);
  }

  public void run(String[] args) {
    if (!parseArgs(args)) {
      return;
    }

    conf = new Configuration();
    conf.set("fs.default.name", "hdfs://localhost:9000/");
    try {
      Configuration.dumpConfiguration(conf, new OutputStreamWriter(System.out));

      System.out.printf("Reading data\n");
      input = IOUtils.getKeysAndVectors(inputFile, projectionDimension);
      reducedVectors = input.getSecond();

      fileOut = new PrintWriter(new FileOutputStream(outputFile));
      fileOut.printf("run, type, time, cluster, distance.mean, distance.sd, count\n");
      for (int i = 0; i < NUM_RUNS; ++i) {
        runInstance(i);
      }
      fileOut.close();
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

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("where to dump the CSV file with the results")
        .create();

    Option projectOption = builder.withLongName("project")
        .withShortName("p")
        .withRequired(true)
        .withDescription("if set, projects the input vectors down to the requested number of dimensions")
        .withArgument(argumentBuilder.withName("project").withMaximum(1).create())
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(outputFileOption)
        .withOption(projectOption)
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

    inputFile = (String)cmdLine.getValue(inputFileOption);
    outputFile = (String)cmdLine.getValue(outputFileOption);
    if (cmdLine.hasOption(projectOption)) {
      projectionDimension = Integer.parseInt((String)cmdLine.getValue(projectOption));
    }
    return true;
  }

  public static void main(String[] args) {
    new ClusterQuality20NewsGroups().run(args);
  }
}
