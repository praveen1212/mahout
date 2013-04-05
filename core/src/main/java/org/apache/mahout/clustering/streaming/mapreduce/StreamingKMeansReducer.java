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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.streaming.cluster.BallKMeans;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.apache.mahout.math.random.WeightedThing;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

public class StreamingKMeansReducer extends Reducer<IntWritable, CentroidWritable, IntWritable,
    CentroidWritable> {

  private Configuration conf;

  private double trainTestSplit;

  private static final Logger log = LoggerFactory.getLogger(StreamingKMeansReducer.class);

  @Override
  public void setup(Context context) {
    // At this point the configuration received from the Driver is assumed to be valid.
    // No other checks are made.
    conf = context.getConfiguration();
  }

  @Override
  public void reduce(IntWritable key, Iterable<CentroidWritable> centroids,
                     Context context) throws IOException, InterruptedException {
    Pair<List<Centroid>, List<Centroid>> splitCentroids = splitTrainTestRaw(centroids, conf);
    int index = 0;
    for (Vector centroid : getBestCentroids(splitCentroids.getFirst(), splitCentroids.getSecond(), conf)) {
      context.write(new IntWritable(index), new CentroidWritable((Centroid)centroid));
      ++index;
    }
  }

  public static Pair<List<Centroid>, List<Centroid>> splitTrainTestRaw(Iterable<CentroidWritable> centroids,
                                                                       Configuration conf) {
    // New lists must be created because Hadoop iterators mutate the contents of the Writable in
    // place, without allocating new references when iterating through the centroids Iterable.
    return splitTrainTest(Iterables.transform(centroids, new Function<CentroidWritable, Centroid>() {
      @Override
      public Centroid apply(CentroidWritable input) {
        Preconditions.checkNotNull(input);
        return input.getCentroid().clone();
      }
    }), conf);
  }

  public static Pair<List<Centroid>, List<Centroid>> splitTrainTest(Iterable<Centroid> centroids,
                                                                    Configuration conf) {
    // We split the incoming centroids in two groups for training and testing.
    // The idea is that we want to see how well our clusters model the distribution and so for this we only
    // "train" (adjust them) on the training set and see how well the centroids we get fit the test points.
    // We're using the exact same algorithm, but since it's randomized, different restarts could help.
    //
    // The cost that we're trying to minimize is the sum of all the distances from each training point to
    // its closest centroid.
    float trainTestSplit = conf.getFloat(StreamingKMeansDriver.TRAIN_TEST_SPLIT, 0.9f);
    List<Centroid> trainIntermediateCentroids = Lists.newArrayList();
    List<Centroid> testIntermediateCentroids = Lists.newArrayList();
    for (Centroid currCentroid : centroids) {
      if (RandomUtils.nextDouble() <= trainTestSplit) {
        trainIntermediateCentroids.add(currCentroid);
      } else {
        testIntermediateCentroids.add(currCentroid);
      }
    }
    log.info("Split data set into {} training vectors and {} test vectors",
        trainIntermediateCentroids.size(), testIntermediateCentroids.size());
    return new Pair<List<Centroid>, List<Centroid>>(trainIntermediateCentroids, testIntermediateCentroids);
  }

  public static Iterable<Vector> getBestCentroids(List<Centroid> trainIntermediateCentroids,
                                                  List<Centroid> testIntermediateCentroids, Configuration conf) {
    int numClusters = conf.getInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, 1);
    int maxNumIterations = conf.getInt(StreamingKMeansDriver.MAX_NUM_ITERATIONS, 10);
    int numRuns = conf.getInt(StreamingKMeansDriver.NUM_BALLKMEANS_RUNS, 10);

    // Run multiple BallKMeans run picking the one with the best cost.
    UpdatableSearcher bestSearcher = null;
    // List<Centroid> bestCentroids = Lists.newArrayListWithExpectedSize(numClusters);
    double bestCost = Double.MAX_VALUE;
    double worstCost = 0;
    for (int i = 0; i < numRuns; ++i) {
      BallKMeans clusterer = new BallKMeans(StreamingKMeansUtilsMR.searcherFromConfiguration(conf, log),
          numClusters, maxNumIterations);
      UpdatableSearcher currSearcher = clusterer.cluster(trainIntermediateCentroids);
      boolean emptyCluster = false;
      double totalCost = 0;
      for (Centroid testCentroid : testIntermediateCentroids) {
        List<WeightedThing<Vector>> closest = currSearcher.search(testCentroid, 1);
        totalCost += closest.get(0).getWeight();
      }
      if (emptyCluster) {
        continue;
      }
      if (totalCost < bestCost) {
        bestCost = totalCost;
        bestSearcher = currSearcher;
      }
      if (totalCost > worstCost) {
        worstCost = totalCost;
      }
    }
    log.info("After {} runs, worst cost {}, best cost {}", numRuns, worstCost, bestCost);
    // Since the test points were not used to compute the Centroids, they're not part of the weight.
    // This is a problem for applications where the weight of a cluster needs to be more precise.
    for (Centroid centroid : testIntermediateCentroids) {
      List<WeightedThing<Vector>> closest = bestSearcher.search(centroid, 1);
      ((Centroid)(closest.get(0).getValue())).addWeight(centroid.getWeight());
    }
    return bestSearcher;
  }
}
