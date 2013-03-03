package org.apache.mahout.clustering.streaming.search;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.apache.mahout.clustering.streaming.LumpyData;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.random.WeightedThing;
import org.junit.Test;

import java.util.List;

public class SearchQualityTest {
  private static final int NUM_DATA_POINTS = 1 << 13;
  private static final int NUM_QUERIES = 1 << 6;
  private static final int NUM_DIMENSIONS = 40;
  private static final int NUM_PROJECTIONS = 3;
  private static final int SEARCH_SIZE = 10;

  public static Matrix lumpyRandomData(int numDataPoints, int numDimensions) {
    final Matrix data = new DenseMatrix(numDataPoints, numDimensions);
    final LumpyData clusters = new LumpyData(numDimensions, 0.05, 10);
    for (MatrixSlice row : data) {
      row.vector().assign(clusters.sample());
    }
    return data;
  }

  @Test
  public void testOverlapAndRuntime() {
    Matrix dataPoints = lumpyRandomData(NUM_DATA_POINTS, NUM_DIMENSIONS);
    Matrix queries = lumpyRandomData(NUM_QUERIES, NUM_DIMENSIONS);

    Searcher bruteSearcher = new BruteSearch(new EuclideanDistanceMeasure());
    bruteSearcher.addAll(dataPoints);
    Pair<List<List<WeightedThing<Vector>>>, Long> reference = getResultsAndRuntime(bruteSearcher, queries);

    List<Searcher> searchers = Lists.newArrayList();
    searchers.add(new ProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE));
    searchers.add(new FastProjectionSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS, SEARCH_SIZE));
    searchers.add(new LocalitySensitiveHashSearch(new EuclideanDistanceMeasure(), NUM_PROJECTIONS));

    System.out.printf("BruteSearch: avg_time(1 query) %f[s]\n", reference.getSecond() / (queries.numRows() * 1.0));

    for (Searcher searcher : searchers) {
      searcher.addAll(dataPoints);
      Pair<List<List<WeightedThing<Vector>>>, Long> results = getResultsAndRuntime(searcher, queries);
      int numFirstMatches = 0;
      int numMatches = 0;
      for (int i = 0; i < queries.numRows(); ++i) {
        if (reference.getFirst().get(i).get(0).getValue().equals(
            results.getFirst().get(i).get(0).getValue())) {
          ++numFirstMatches;
        }
        for (Vector v : Iterables.transform(reference.getFirst().get(i), new StripWeight())) {
          for (Vector w : Iterables.transform(results.getFirst().get(i), new StripWeight())) {
            if (v.equals(w)) {
              ++numMatches;
            }
          }
        }
      }
      System.out.printf("%s [%d]: first %d; total %d; avg_time(1 query) %f[s]\n",
          searcher.getClass().getName(), queries.numRows(),
          numFirstMatches, numMatches, results.getSecond() / (queries.numRows() * 1.0));
    }
  }

  public Pair<List<List<WeightedThing<Vector>>>, Long> getResultsAndRuntime(Searcher searcher,
                                                                            Iterable<? extends Vector> queries) {
    long start = System.currentTimeMillis();
    List<List<WeightedThing<Vector>>> results = searcher.search(queries, 10);
    long end = System.currentTimeMillis();
    return new Pair<List<List<WeightedThing<Vector>>>, Long>(results, end - start);
  }

  static class StripWeight implements Function<WeightedThing<Vector>, Vector> {
    @Override
    public Vector apply(WeightedThing<Vector> input) {
      Preconditions.checkArgument(input != null);
      //noinspection ConstantConditions
      return input.getValue();
    }
  }
}
