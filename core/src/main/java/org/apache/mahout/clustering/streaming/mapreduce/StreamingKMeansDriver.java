/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.clustering.streaming.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.neighborhood.BruteSearch;
import org.apache.mahout.math.neighborhood.LocalitySensitiveHashSearch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Classifies the vectors into different clusters found by the clustering
 * algorithm.
 */
public final class StreamingKMeansDriver extends AbstractJob {
  /**
   *
   */
  public static final String ESTIMATED_NUM_MAP_CLUSTERS = "estimatedNumMapClusters";
  /**
   * The Searcher class when performing nearest neighbor search in StreamingKMeans.
   * Defaults to BruteSearch.
   */
  public static final String SEARCHER_CLASS_OPTION = "searcherClass";
  /**
   * The number of projections to use when using a projection searcher like ProjectionSearch or
   * FastProjectionSearch. Projection searches work by projection the all the vectors on to a set of
   * basis vectors and searching for the projected query in that totally ordered set. This
   * however can produce false positives (vectors that are closer when projected than they would
   * actually be.
   * So, there must be more than one projection vectors in the basis. This variable is the number
   * of vectors in a basis.
   * Defaults to 20.
   */
  public static final String NUM_PROJECTIONS_OPTION = "numProjections";
  /**
   * When using approximate searches (anything that's not BruteSearch),
   * more than just the seemingly closest element must be considered. This variable has different
   * meanings depending on the actual Searcher class used but is a measure of how many candidates
   * will be considered.
   * See the ProjectionSearch, FastProjectionSearch, LocalitySensitiveHashSearch classes for more
   * details.
   * Defaults to 10.
   */
  public static final String SEARCH_SIZE_OPTION = "searchSize";
  /**
   * After mapping finishes, we get an intermediate set of vectors that represent approximate
   * clusterings of the data from each Mapper. These can be clustered by the Reducer using
   * BallKMeans in memory. This variable is the maximum number of iterations in the final
   * BallKMeans algorithm.
   * Defaults to 10.
   */
  public static final String MAX_NUM_ITERATIONS = "maxNumIterations";
  /**
   * The initial estimated distance cutoff between two points for forming new clusters.
   * @see org.apache.mahout.clustering.streaming.cluster.StreamingKMeans
   * Defaults to 10e-6.
   */
  public static final String ESTIMATED_DISTANCE_CUTOFF = "estimatedDistanceCutoff";

  /**
   * The percentage of points that go into the "training" set when evaluating BallKMeans runs in the reducer.
   */
  public static final String TRAIN_TEST_SPLIT = "trainTestSplit";

  /**
   * The percentage of points that go into the "training" set when evaluating BallKMeans runs in the reducer.
   */
  public static final String NUM_BALLKMEANS_RUNS = "numBallKMeansRuns";

  private static final Logger log = LoggerFactory.getLogger(StreamingKMeansDriver.class);

  @Override
  public int run(String[] args) throws Exception {
    // Standard options for any Mahout job.
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());

    // The number of clusters to create for the data.
    addOption(DefaultOptionCreator.numClustersOption().withDescription(
        "The k in k-Means. Approximately this many clusters will be generated.").create());

    /* StreamingKMeans (mapper) options */

    // There will be k final clusters, but in the Map phase to get a good approximation of the data, O(k log n)
    // clusters are needed. Since n is the number of data points and not knowable until reading all the vectors,
    // provide a decent estimate.
    addOption(ESTIMATED_NUM_MAP_CLUSTERS, "km", "The estimated number of clusters to use for the " +
        "Map phase of the job when running StreamingKMeans. This should be around k * log(n), " +
        "where k is the final number of clusters and n is the total number of data points to " +
        "cluster.");

    addOption(ESTIMATED_DISTANCE_CUTOFF, "e", "The initial estimated distance cutoff between two " +
        "points for forming new clusters", "1e-6");

    /* BallKMeans (reducer) options */

    addOption(MAX_NUM_ITERATIONS, "mi", "The maximum number of iterations to run for the " +
        "BallKMeans algorithm used by the reducer.", "10");

    addOption(TRAIN_TEST_SPLIT, "trte", "A double value between 0 and 1 that represents the percentage of " +
        "points to be used for 'training' and for 'testing' different clustering runs in the final BallKMeans " +
        "step. If no value is given, defaults to 0.9", "0.9");

    addOption(NUM_BALLKMEANS_RUNS, "nbkm", "Number of BallKMeans runs to use at the end to try to cluster the " +
        "points. If no value is given, defaults to 10", "10");

    /* Nearest neighbor search options */

    // The distance measure used for computing the distance between two points. Generally, the
    // SquaredEuclideanDistance is used for clustering problems (it's equivalent to CosineDistance for normalized
    // vectors).
    // WARNING! You can use any metric but most of the literature is for the squared euclidean distance.
    addOption(DefaultOptionCreator.distanceMeasureOption().create());

    // The default searcher should be something more efficient that BruteSearch (ProjectionSearch, ...). See
    // o.a.m.math.neighborhood.*
    addOption(SEARCHER_CLASS_OPTION, "sc", "The type of searcher to be used when performing nearest " +
        "neighbor searches. Defaults to BruteSearch.", BruteSearch.class.getCanonicalName());

    // In the original paper, the authors used 1 projection vector.
    addOption(NUM_PROJECTIONS_OPTION, "np", "The number of projections considered in estimating the " +
        "distances between vectors. Only used when the distance measure requested is either " +
        "ProjectionSearch or FastProjectionSearch. If no value is given, defaults to 5.", "5");

    addOption(SEARCH_SIZE_OPTION, "s", "In more efficient searches (non BruteSearch), " +
        "not all distances are calculated for determining the nearest neighbors. The number of " +
        "elements whose distances from the query vector is actually computer is proportional to " +
        "searchSize. If no value is given, defaults to 3.", "3");

    if (parseArguments(args) == null) {
      return -1;
    }
    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    configureOptionsForWorkers();
    run(getConf(), input, output);
    return 0;
  }

  private void configureOptionsForWorkers() throws ClassNotFoundException, IllegalAccessException,
      InstantiationException {
    log.info("Starting to configure options for workers");

    String numClustersStr = getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION);
    int numClusters = Integer.parseInt(numClustersStr);

    // StreamingKMeans
    String estimatedNumMapClustersStr = getOption(ESTIMATED_NUM_MAP_CLUSTERS);
    int estimatedNumMapClusters = Integer.parseInt(estimatedNumMapClustersStr);

    String estimatedDistanceCutoffStr = getOption(ESTIMATED_DISTANCE_CUTOFF);
    float estimatedDistanceCutoff = Float.parseFloat(estimatedDistanceCutoffStr);

    // BallKMeans
    String maxNumIterationsStr = getOption(MAX_NUM_ITERATIONS);
    int maxNumIterations = Integer.parseInt(maxNumIterationsStr);

    String trainTestSplitStr = getOption(TRAIN_TEST_SPLIT);
    float trainTestSplit = Float.parseFloat(trainTestSplitStr);

    String numBallKMeansRunsStr = getOption(NUM_BALLKMEANS_RUNS);
    int numBallKMeansRuns = Integer.parseInt(numBallKMeansRunsStr);

    // Nearest neighbor search
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = EuclideanDistanceMeasure.class.getName();
      log.info("No measure class given, using EuclideanDistanceMeasure");
    }

    String searcherClass = getOption(SEARCHER_CLASS_OPTION);
    // Get more parameters depending on the kind of search class we're working with. BruteSearch
    // doesn't need anything else.
    // LocalitySensitiveHashSearch and ProjectionSearches need searchSize.
    // ProjectionSearches also need the number of projections.
    boolean getSearchSize = false;
    boolean getNumProjections = false;
    if (!searcherClass.equals(BruteSearch.class.getName())) {
      getSearchSize = true;
      if (!searcherClass.equals(LocalitySensitiveHashSearch.class.getName())) {
        getNumProjections = true;
      }
    }

    // The search size to use. This is quite fuzzy and might end up not being configurable at all.
    int searchSize = 0;
    if (getSearchSize) {
      String searchSizeStr = getOption(SEARCH_SIZE_OPTION);
      searchSize = Integer.parseInt(searchSizeStr);
    }

    // The number of projections to use. This is only useful in projection searches which
    // project the vectors on multiple basis vectors to get distance estimates that are faster to
    // calculate.
    int numProjections = 0;
    if (getNumProjections) {
      String numProjectionsStr = getOption(NUM_PROJECTIONS_OPTION);
      numProjections = Integer.parseInt(numProjectionsStr);
    }

    configureOptionsForWorkers(getConf(), numClusters,
        /* StreamingKMeans */
        estimatedNumMapClusters,  estimatedDistanceCutoff,
        /* BallKMeans */
        maxNumIterations, trainTestSplit, numBallKMeansRuns,
        /* Searcher */
        measureClass, searcherClass,  searchSize, numProjections);
  }

  /**
   * Checks the parameters for a StreamingKMeans job and prepares a Configuration with them.
   *
   * @param conf the Configuration to populate
   * @param numClusters k, the number of clusters at the end
   * @param estimatedNumMapClusters O(k log n), the number of clusters requested from each mapper
   * @param estimatedDistanceCutoff an estimate of the minimum distance that separates two clusters (can be smaller and
   *                                will be increased dynamically)
   * @param maxNumIterations the maximum number of iterations of BallKMeans
   * @param trainTestSplit the percentage of vectors assigned to the training set for selecting the best final centers
   * @param numBallKMeansRuns the number of BallKMeans runs in the reducer that determine the centroids to return
   *                          (clusters are computed for the training set and the error is computed on the test set)
   * @param measureClass string, name of the distance measure class; theory works for Euclidean-like distances
   * @param searcherClass string, name of the searcher that will be used for nearest neighbor search
   * @param searchSize the number of closest neighbors to look at for selecting the closest one in approximate nearest
   *                   neighbor searches
   * @param numProjections the number of projected vectors to use for faster searching (only useful for ProjectionSearch
   *                       or FastProjectionSearch); @see org.apache.mahout.math.neighborhood.ProjectionSearch
   */
  public static void configureOptionsForWorkers(Configuration conf,
                                                int numClusters,
                                                /* StreamingKMeans */
                                                int estimatedNumMapClusters, float estimatedDistanceCutoff,
                                                /* BallKMeans */
                                                int maxNumIterations, float trainTestSplit, int numBallKMeansRuns,
                                                /* Searcher */
                                                String measureClass, String searcherClass,
                                                int searchSize, int numProjections) {
    // Checking preconditions for the parameters.
    Preconditions.checkArgument(numClusters > 0, "Invalid number of clusters requested");

    // StreamingKMeans
    Preconditions.checkArgument(estimatedNumMapClusters > numClusters, "Invalid number of estimated map " +
        "clusters; There must be more than the final number of clusters (k log n vs k)");
    Preconditions.checkArgument(estimatedDistanceCutoff > 0, "estimatedDistanceCutoff cannot be negative");

    // BallKMeans
    Preconditions.checkArgument(maxNumIterations > 0, "Must have at least one BallKMeans iteration");
    Preconditions.checkArgument(0 < trainTestSplit && trainTestSplit <= 1, "train/test split is not in the interval " +
        "[0, 1)");
    Preconditions.checkArgument(numBallKMeansRuns > 0, "numBallKMeans cannot be negative");

    // Searcher
    if (!searcherClass.contains("Brute")) {
      // These tests only make sense when a relevant searcher is being used.
      Preconditions.checkArgument(searchSize > 0, "Invalid searchSize. Must be positive.");
      if (searcherClass.contains("Projection")) {
        Preconditions.checkArgument(numProjections > 0, "Invalid numProjections. Must be positive");
      }
    }

    // Setting the parameters in the Configuration.
    conf.setInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, numClusters);
    /* StreamingKMeans */
    conf.setInt(ESTIMATED_NUM_MAP_CLUSTERS, estimatedNumMapClusters);
    conf.setFloat(ESTIMATED_DISTANCE_CUTOFF, estimatedDistanceCutoff);
    /* BallKMeans */
    conf.setInt(MAX_NUM_ITERATIONS, maxNumIterations);
    conf.setFloat(TRAIN_TEST_SPLIT, trainTestSplit);
    conf.setInt(NUM_BALLKMEANS_RUNS, numBallKMeansRuns);
    /* Searcher */
    try {
      Class.forName(measureClass);
    }  catch (ClassNotFoundException e) {
      log.error("Measure class not found " + measureClass, e);
    }
    conf.set(DefaultOptionCreator.DISTANCE_MEASURE_OPTION, measureClass);
    try {
      Class.forName(searcherClass);
    } catch (ClassNotFoundException e) {
      log.error("Searcher class not found " + measureClass, e);
    }
    conf.set(SEARCHER_CLASS_OPTION, searcherClass);
    conf.setInt(SEARCH_SIZE_OPTION, searchSize);
    conf.setInt(NUM_PROJECTIONS_OPTION, numProjections);
    log.info("Parameters are: [k] numClusters {}; " +
        "[SKM] estimatedNumMapClusters {}; estimatedDistanceCutoff" +
        "[BKM] maxNumIterations {}; trainTestSplit {}; numBallKMeansRuns {};" +
        "[S] measureClass {}; searcherClass {}; searcherSize {}; numProjections {}; " +
        "maxNumIterations {}", numClusters, estimatedNumMapClusters, estimatedDistanceCutoff,
        maxNumIterations, trainTestSplit, numBallKMeansRuns,
        measureClass, searcherClass, searchSize, numProjections);
  }

  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the results of the final iteration to
   * cluster the input vectors.
   *
   * @param input
   *          the directory pathname for input points
   * @param output
   *          the directory pathname for output points
   */
  public static void run(Configuration conf, Path input, Path output)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("Starting StreamingKMeans clustering for vectors in {}; results are output to {}",
        input.toString(), output.toString());

    // Prepare Job for submission.
    Job job = new Job(conf, "StreamingKMeans");

    // Input and output file format.
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    // Mapper output Key and Value classes.
    // We don't really need to output anything as a key, since there will only be 1 reducer.
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(CentroidWritable.class);

    // Reducer output Key and Value classes.
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(CentroidWritable.class);

    // Mapper and Reducer classes.
    job.setMapperClass(StreamingKMeansMapper.class);
    job.setReducerClass(StreamingKMeansReducer.class);

    // There is only one reducer so that the intermediate centroids get collected on one
    // machine and are clustered in memory to get the right number of clusters.
    job.setNumReduceTasks(1);

    // Set input and output paths for the job.
    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    // Set the JAR (so that the required libraries are available) and run.
    job.setJarByClass(StreamingKMeansDriver.class);

    long start = System.currentTimeMillis();
    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("StreamingKMeans interrupted");
    }
    long end = System.currentTimeMillis();

    log.info("StreamingKMeans clustering complete. Results are in {}. Took {} ms",
        output.toString(), end - start);
  }

  /**
   * Constructor to be used by the ToolRunner.
   */
  private StreamingKMeansDriver() {
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new StreamingKMeansDriver(), args);
  }
}
