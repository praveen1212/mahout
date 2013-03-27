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

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mrunit.mapreduce.MapDriver;
import org.apache.hadoop.mrunit.mapreduce.MapReduceDriver;
import org.apache.hadoop.mrunit.mapreduce.ReduceDriver;
import org.apache.mahout.clustering.streaming.cluster.ClusteringUtils;
import org.apache.mahout.clustering.streaming.cluster.DataUtils;
import org.apache.mahout.clustering.streaming.cluster.StreamingKMeans;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.neighborhood.BruteSearch;
import org.apache.mahout.math.neighborhood.LocalitySensitiveHashSearch;
import org.apache.mahout.math.neighborhood.ProjectionSearch;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

@RunWith(value = Parameterized.class)
public class StreamingKMeansTestMR {
  private static final int NUM_DATA_POINTS = 1 << 15;
  private static final int NUM_DIMENSIONS = 8;
  private static final int NUM_PROJECTIONS = 3;
  private static final int SEARCH_SIZE = 5;
  private static final int MAX_NUM_ITERATIONS = 10;
  private static final double DISTANCE_CUTOFF = 1e-10;

  private static Pair<List<Centroid>, List<Centroid>> syntheticData =
      DataUtils.sampleMultiNormalHypercube(NUM_DIMENSIONS, NUM_DATA_POINTS, 1e-5);

  private Configuration configuration;
  private Logger log = LoggerFactory.getLogger(StreamingKMeansTestMR.class);

  public StreamingKMeansTestMR(String searcherClassName, String distanceMeasureClassName) {
    configuration = new Configuration();
    configuration.set(DefaultOptionCreator.DISTANCE_MEASURE_OPTION, distanceMeasureClassName);
    configuration.setInt(StreamingKMeansDriver.SEARCH_SIZE_OPTION, SEARCH_SIZE);
    configuration.setInt(StreamingKMeansDriver.NUM_PROJECTIONS_OPTION, NUM_PROJECTIONS);
    configuration.set(StreamingKMeansDriver.SEARCHER_CLASS_OPTION, searcherClassName);
    configuration.setInt(DefaultOptionCreator.NUM_CLUSTERS_OPTION, 1 << NUM_DIMENSIONS);
    configuration.setInt(StreamingKMeansDriver.ESTIMATED_NUM_MAP_CLUSTERS,
        (1 << NUM_DIMENSIONS) * (int)Math.log(NUM_DATA_POINTS));
    configuration.setFloat(StreamingKMeansDriver.ESTIMATED_DISTANCE_CUTOFF, (float) DISTANCE_CUTOFF);
    configuration.setInt(StreamingKMeansDriver.MAX_NUM_ITERATIONS, MAX_NUM_ITERATIONS);
  }

  @Parameterized.Parameters
  public static List<Object[]> generateData() {
    return Arrays.asList(new Object[][]{
        {ProjectionSearch.class.getName(), SquaredEuclideanDistanceMeasure.class.getName()},
        // {FastProjectionSearch.class.getName(), SquaredEuclideanDistanceMeasure.class.getName()},
        {LocalitySensitiveHashSearch.class.getName(), SquaredEuclideanDistanceMeasure.class.getName()}
    });
  }

  @Test
  public void testHypercubeMapper() throws IOException {
    System.out.printf("%s mapper test\n", configuration.get(StreamingKMeansDriver.SEARCHER_CLASS_OPTION));
    MapDriver<Writable, VectorWritable, IntWritable, CentroidWritable> mapDriver =
        MapDriver.newMapDriver(new StreamingKMeansMapper());
    mapDriver.setConfiguration(configuration);
    for (Centroid datapoint : syntheticData.getFirst()) {
      mapDriver.addInput(new IntWritable(0), new VectorWritable(datapoint));
    }
    List<org.apache.hadoop.mrunit.types.Pair<IntWritable,CentroidWritable>> results = mapDriver.run();
    BruteSearch resultSearcher = new BruteSearch(new SquaredEuclideanDistanceMeasure());
    for (org.apache.hadoop.mrunit.types.Pair<IntWritable, CentroidWritable> result : results) {
      resultSearcher.add(result.getSecond().getCentroid());
    }
    System.out.printf("Clustered the data into %d clusters\n", results.size());
    for (Vector mean : syntheticData.getSecond()) {
      WeightedThing<Vector> closest = resultSearcher.search(mean, 1).get(0);
      assertThat("Weight " + closest.getWeight() + " not less than 0.5", closest.getWeight(), lessThan(0.5));
    }
  }

  @Test
  public void testMapperVsLocal() throws IOException {
    System.out.printf("%s mapper vs local test\n", configuration.get(StreamingKMeansDriver.SEARCHER_CLASS_OPTION));

    // Clusters the data using the StreamingKMeansMapper.
    MapDriver<Writable, VectorWritable, IntWritable, CentroidWritable> mapDriver =
        MapDriver.newMapDriver(new StreamingKMeansMapper());
    mapDriver.setConfiguration(configuration);
    for (Centroid datapoint : syntheticData.getFirst()) {
      mapDriver.addInput(new IntWritable(0), new VectorWritable(datapoint));
    }
    List<Centroid> mapperCentroids = Lists.newArrayList();
    for (org.apache.hadoop.mrunit.types.Pair<IntWritable, CentroidWritable> pair : mapDriver.run()) {
      mapperCentroids.add(pair.getSecond().getCentroid());
    }

    // Clusters the data using local batch StreamingKMeans.
    StreamingKMeans batchClusterer = new StreamingKMeans(StreamingKMeansUtilsMR
        .searcherFromConfiguration(configuration, log),
        (1 << NUM_DIMENSIONS) * (int)Math.log(NUM_DATA_POINTS), DISTANCE_CUTOFF);
    batchClusterer.cluster(syntheticData.getFirst());
    List<Centroid> batchCentroids = Lists.newArrayList();
    for (Vector v : batchClusterer) {
      batchCentroids.add((Centroid) v);
    }

    // Clusters the data using point by point StreamingKMeans.
    StreamingKMeans perPointClusterer = new StreamingKMeans(StreamingKMeansUtilsMR
        .searcherFromConfiguration(configuration, log),
        (1 << NUM_DIMENSIONS) * (int)Math.log(NUM_DATA_POINTS), DISTANCE_CUTOFF);
    for (Centroid datapoint : syntheticData.getFirst()) {
      perPointClusterer.cluster(datapoint);
    }
    List<Centroid> perPointCentroids = Lists.newArrayList();
    for (Vector v : perPointClusterer) {
      perPointCentroids.add((Centroid) v);
    }

    // Computes the cost (total sum of distances) of these different clusterings.
    double mapperCost = ClusteringUtils.totalClusterCost(syntheticData.getFirst(), mapperCentroids);
    double localCost = ClusteringUtils.totalClusterCost(syntheticData.getFirst(), batchCentroids);
    double perPointCost = ClusteringUtils.totalClusterCost(syntheticData.getFirst(), perPointCentroids);
    System.out.printf("[Total cost] Mapper %f [%d] Local %f [%d] Perpoint local %f [%d];" +
        "[ratio m-vs-l %f] [ratio pp-vs-l %f]\n", mapperCost, mapperCentroids.size(),
        localCost, batchCentroids.size(), perPointCost, perPointCentroids.size(),
        mapperCost / localCost, perPointCost / localCost);

    // These ratios should be close to 1.0 and have been observed to be go as low as 0.6 and as low as 1.5.
    // A buffer of [0.2, 1.8] seems appropriate.
    assertEquals("Mapper StreamingKMeans / Batch local StreamingKMeans total cost ratio too far from 1",
        mapperCost / localCost, 1.0, 0.8);
    assertEquals("One by one local StreamingKMeans / Batch local StreamingKMeans total cost ratio too high",
        perPointCost / localCost, 1.0, 0.8);
  }

  @Test
  public void testHypercubeReducer() throws IOException {
    System.out.printf("%s reducer test\n", configuration.get(StreamingKMeansDriver.SEARCHER_CLASS_OPTION));
    StreamingKMeans clusterer = new StreamingKMeans(StreamingKMeansUtilsMR
        .searcherFromConfiguration(configuration, log),
        (1 << NUM_DIMENSIONS) * (int)Math.log(NUM_DATA_POINTS), DISTANCE_CUTOFF);
    long start = System.currentTimeMillis();
    clusterer.cluster(syntheticData.getFirst());
    long end = System.currentTimeMillis();
    System.out.printf("%f [s]\n", (end - start) / 1000.0);
    ReduceDriver<IntWritable, CentroidWritable, IntWritable, CentroidWritable> reduceDriver =
        ReduceDriver.newReduceDriver(new StreamingKMeansReducer());
    reduceDriver.setConfiguration(configuration);
    List<CentroidWritable> reducerInputs = Lists.newArrayList();
    int postMapperTotalWeight = 0;
    for (Centroid intermediateCentroid : clusterer) {
      reducerInputs.add(new CentroidWritable(intermediateCentroid));
      postMapperTotalWeight += intermediateCentroid.getWeight();
    }
    reduceDriver.addInput(new IntWritable(0), reducerInputs);
    List<org.apache.hadoop.mrunit.types.Pair<IntWritable, CentroidWritable>> results =
        reduceDriver.run();
    testReducerResults(postMapperTotalWeight, results);
  }

  @Test
  public void testHypercubeMapReduce() throws IOException {
    System.out.printf("%s full test\n", configuration.get(StreamingKMeansDriver.SEARCHER_CLASS_OPTION));
    MapReduceDriver<Writable, VectorWritable, IntWritable, CentroidWritable, IntWritable, CentroidWritable>
        mapReduceDriver = new MapReduceDriver<Writable, VectorWritable, IntWritable, CentroidWritable,
        IntWritable, CentroidWritable>(new StreamingKMeansMapper(), new StreamingKMeansReducer());
    mapReduceDriver.setConfiguration(configuration);
    for (Centroid datapoint : syntheticData.getFirst()) {
      mapReduceDriver.addInput(new IntWritable(0), new VectorWritable(datapoint));
    }
    List<org.apache.hadoop.mrunit.types.Pair<IntWritable, CentroidWritable>> results = mapReduceDriver.run();
    testReducerResults(syntheticData.getFirst().size(), results);
  }

  private void testReducerResults(int totalWeight, List<org.apache.hadoop.mrunit.types.Pair<IntWritable,
      CentroidWritable>> results) {
    int expectedNumClusters = 1 << NUM_DIMENSIONS;
    double expectedWeight = totalWeight / expectedNumClusters;
    int numClusters = 0;
    int numUnbalancedClusters = 0;
    int totalReducerWeight = 0;
    for (org.apache.hadoop.mrunit.types.Pair<IntWritable, CentroidWritable> result : results) {
      if (result.getSecond().getCentroid().getWeight() != expectedWeight) {
        System.out.printf("Unbalanced weight %f in centroid %d\n",  result.getSecond().getCentroid().getWeight(),
            result.getSecond().getCentroid().getIndex());
        ++numUnbalancedClusters;
      }
      assertEquals("Final centroid index is invalid", numClusters, result.getFirst().get());
      totalReducerWeight += result.getSecond().getCentroid().getWeight();
      ++numClusters;
    }
    System.out.printf("%d clasters are unbalanced\n", numUnbalancedClusters);
    assertEquals("Invalid total weight", totalWeight, totalReducerWeight);
    assertEquals("Invalid number of clusters", 1 << NUM_DIMENSIONS, numClusters);
  }

}
