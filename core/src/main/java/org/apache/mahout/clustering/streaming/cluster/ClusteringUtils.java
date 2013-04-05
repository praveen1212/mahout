package org.apache.mahout.clustering.streaming.cluster;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
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
    UpdatableSearcher searcher = new ProjectionSearch(distanceMeasure, 3, 1);
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
    UpdatableSearcher searcher = new ProjectionSearch(distanceMeasure, 3, 1);
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
   * @param data the datapoints whose distance is to be estimated.
   * @param distanceMeasure the distance measure used to compute the distance between two points.
   * @return the minimum distance between the first sampleLimit points
   * @see org.apache.mahout.clustering.streaming.cluster.StreamingKMeans#clusterInternal(Iterable, boolean)
   */
  public static double estimateDistanceCutoff(Iterable<? extends Vector> data,
                                              DistanceMeasure distanceMeasure, int sampleLimit) {
    ProjectionSearch searcher = new ProjectionSearch(distanceMeasure, 3, 1);
    searcher.addAll(data);
    double minDistance = Double.POSITIVE_INFINITY;
    for (Vector u : data) {
      if (sampleLimit == 0) {
        break;
      }
      double closest = searcher.search(u, 2).get(1).getWeight();
      if (closest < minDistance) {
        minDistance = closest;
      }
      --sampleLimit;
    }
    return minDistance;
  }

  public static double estimateDistanceCutoff(Iterable<? extends Vector> data,
                                              DistanceMeasure distanceMeasure) {
    return estimateDistanceCutoff(data, distanceMeasure, Integer.MAX_VALUE);
  }

  public static double daviesBouldinIndex(List<Centroid> centroids, DistanceMeasure distanceMeasure,
                                          List<OnlineSummarizer> clusterDistanceSummaries) {
    Preconditions.checkArgument(centroids.size() == clusterDistanceSummaries.size(),
        "Number of centroids and cluster summaries differ.");
    int n = centroids.size();
    double totalDBIndex = 0;
    // The inner loop shouldn't be reduced for j = i + 1 to n because the computation of the Davies-Bouldin
    // index is not really symmetric.
    // For a given cluster i, we look for a cluster j that maximizes the ratio of the sum of average distances
    // from points in cluster i to its center and and points in cluster j to its center to the distance between
    // cluster i and cluster j.
    // The maximization is the key issue, as the cluster that maximizes this ratio might be j for i but is NOT
    // NECESSARILY i for j.
    for (int i = 0; i < n; ++i) {
      double averageDistanceI = clusterDistanceSummaries.get(i).getMean();
      double maxDBIndex = 0;
      for (int j = 0; j < n; ++j) {
        if (i == j) {
          continue;
        }
        double dbIndex = (averageDistanceI + clusterDistanceSummaries.get(j).getMean()) /
            distanceMeasure.distance(centroids.get(i), centroids.get(j));
        if (dbIndex > maxDBIndex) {
          maxDBIndex = dbIndex;
        }
      }
      totalDBIndex += maxDBIndex;
    }
    return totalDBIndex / n;
  }

  public static double dunnIndex(List<Centroid> centroids, DistanceMeasure distanceMeasure,
                                 List<OnlineSummarizer> clusterDistanceSummaries) {
    Preconditions.checkArgument(centroids.size() == clusterDistanceSummaries.size(),
        "Number of centroids and cluster summaries differ.");
    int n = centroids.size();
    // Intra-cluster distances will come from the OnlineSummarizer, and will be the median distance (noting that
    // the median for just one value is that value).
    // A variety of metrics can be used for the intra-cluster distance including max distance between two points,
    // mean distance, etc. Median distance was chosen as this is more robust to outliers and characterizes the
    // distribution of distances (from a point to the center) better.
    double maxIntraClusterDistance = 0;
    for (OnlineSummarizer summarizer : clusterDistanceSummaries) {
      if (summarizer.getCount() == 0) {
        continue;
      }
      double intraClusterDistance;
      if (summarizer.getCount() == 1) {
        intraClusterDistance = summarizer.getMean();
      } else {
        intraClusterDistance = summarizer.getMedian();
      }
      if (maxIntraClusterDistance < intraClusterDistance) {
        maxIntraClusterDistance = intraClusterDistance;
      }
    }
    double minDunnIndex = Double.POSITIVE_INFINITY;
    for (int i = 0; i < n; ++i) {
      // Distances are symmetric, so d(i, j) = d(j, i).
      for (int j = i + 1; j < n; ++j) {
        double dunnIndex = distanceMeasure.distance(centroids.get(i), centroids.get(j));
        if (minDunnIndex > dunnIndex) {
          minDunnIndex = dunnIndex;
        }
      }
    }
    return minDunnIndex / maxIntraClusterDistance;
  }
}
