package org.apache.mahout.clustering.streaming.cluster;

import java.util.concurrent.Callable;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.streaming.mapreduce.StreamingKMeansDriver;
import org.apache.mahout.clustering.streaming.mapreduce.StreamingKMeansUtilsMR;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StreamingKMeansThread implements Callable<Iterable<Centroid>> {
  private Configuration conf;
  private Iterable<Centroid> datapoints;

  private static final Logger log = LoggerFactory.getLogger(StreamingKMeansThread.class);

  public StreamingKMeansThread(Path input, Configuration conf) {
    this.datapoints = getCentroidIterable(new SequenceFileValueIterable<VectorWritable>(
        input, false, conf));
    this.conf = conf;
  }

  public StreamingKMeansThread(Iterable<Centroid> datapoints, Configuration conf) {
    this.datapoints = datapoints;
    this.conf = conf;
  }

  @Override
  public Iterable<Centroid> call() throws Exception {
    UpdatableSearcher searcher = StreamingKMeansUtilsMR.searcherFromConfiguration(conf, log);
    int numClusters = conf.getInt(StreamingKMeansDriver.ESTIMATED_NUM_MAP_CLUSTERS, 1);

    System.out.printf("Starting SKM\n");
    /*
    double estimateDistanceCutoff = ClusteringUtils.estimateDistanceCutoff(datapoints,
            StreamingKMeansUtilsMR.searcherFromConfiguration(conf, log).getDistanceMeasure(), 4096);
            */
    double estimateDistanceCutoff = conf.getFloat(StreamingKMeansDriver.ESTIMATED_DISTANCE_CUTOFF, 1e-6f);
    System.out.printf("estDstCtf %f\n", estimateDistanceCutoff);

    StreamingKMeans clusterer = new StreamingKMeans(searcher, numClusters, estimateDistanceCutoff);

    return Iterables.transform(clusterer.cluster(datapoints), new Function<Vector, Centroid>() {
      @Override
      public Centroid apply(Vector input) {
        Preconditions.checkNotNull(input);
        return (Centroid)input;
      }
    });
  }

  public Iterable<Centroid> getCentroidIterable(Iterable<VectorWritable> inputIterable) {
    return Iterables.transform(inputIterable, new Function<VectorWritable, Centroid>() {
      int numVectors = 0;
      @Override
      public Centroid apply(VectorWritable input) {
        Preconditions.checkNotNull(input);
        return new Centroid(numVectors++, new RandomAccessSparseVector(input.get()), 1);
      }
    });
  }
}
