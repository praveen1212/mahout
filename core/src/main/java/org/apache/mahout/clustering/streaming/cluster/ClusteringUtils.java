package org.apache.mahout.clustering.streaming.cluster;

import com.google.common.collect.Lists;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.neighborhood.BruteSearch;
import org.apache.mahout.math.neighborhood.ProjectionSearch;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.apache.mahout.math.stats.OnlineSummarizer;

import java.util.List;

public class ClusteringUtils {
  /**
   * Computes the summaries for the distances in each cluster.
   * @param datapoints iterable of datapoints.
   * @param centroids iterable of Centroids.
   * @return a list of OnlineSummarizers where the i-th element is the summarizer corresponding to the cluster whose
   * index is i.
   */
  public static List<OnlineSummarizer> summarizeClusterDistances(Iterable<? extends Vector> datapoints,
                                                                 Iterable<Centroid> centroids,
                                                                 DistanceMeasure distanceMeasure) {
    UpdatableSearcher searcher = new BruteSearch(distanceMeasure);
    searcher.addAll(centroids);
    List<OnlineSummarizer> summarizers = Lists.newArrayList();
    if (searcher.size() == 0) {
      return summarizers;
    }
    for (int i = 0; i < searcher.size(); ++i) {
      summarizers.add(new OnlineSummarizer());
    }
    for (Vector v : datapoints) {
      Centroid closest = (Centroid)searcher.search(v,  1).get(0).getValue();
      OnlineSummarizer summarizer = summarizers.get(closest.getIndex());
      summarizer.add(distanceMeasure.distance(v, closest));
    }
    return summarizers;
  }

  /**
   * Adds up the distances from each point to its closest cluster and returns the sum.
   * @param datapoints iterable of datapoints.
   * @param centroids iterable of Centroids.
   * @return the total cost described above.
   */
  public static double totalClusterCost(Iterable<? extends Vector> datapoints, Iterable<Centroid> centroids) {
    DistanceMeasure distanceMeasure = new EuclideanDistanceMeasure();
    UpdatableSearcher searcher = new BruteSearch(distanceMeasure);
    searcher.addAll(centroids);
    double totalCost = 0;
    for (Vector v : datapoints) {
      Centroid closest = (Centroid)searcher.search(v, 1).get(0).getValue();
      totalCost += closest.getWeight();
    }
    return totalCost;
  }

  /**
   * Estimates the distance cutoff. In StreamingKMeans, the distance between two vectors divided
   * by this value is used as a probability threshold when deciding whether to form a new cluster
   * or not.
   * Small values (comparable to the minimum distance between two points) are preferred as they
   * guarantee with high likelihood that all but very close points are put in separate clusters
   * initially. The clusters themselves are actually collapsed periodically when their number goes
   * over the maximum number of clusters and the distanceCutoff is increased.
   * So, the returned value is only an initial estimate.
   * @param data
   * @param distanceMeasure
   * @return the minimum distance between the first sampleLimit points
   * @see org.apache.mahout.clustering.streaming.cluster.StreamingKMeans#clusterInternal(Iterable, boolean)
   */
  public static double estimateDistanceCutoff(Iterable<? extends Vector> data,
                                              DistanceMeasure distanceMeasure) {
    ProjectionSearch searcher = new ProjectionSearch(distanceMeasure, 3, 1);
    searcher.addAll(data);
    double minDistance = Double.POSITIVE_INFINITY;
    for (Vector u : data) {
      double closest = searcher.search(u, 2).get(1).getWeight();
      if (closest < minDistance) {
        minDistance = closest;
      }
    }
    return minDistance;
  }
}
