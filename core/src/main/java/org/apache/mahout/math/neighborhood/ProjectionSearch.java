/*
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

package org.apache.mahout.math.neighborhood;

import com.google.common.base.Preconditions;
import com.google.common.collect.*;
import org.apache.mahout.clustering.streaming.cluster.RandomProjector;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.*;
import org.apache.mahout.math.random.WeightedThing;

import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

/**
 * Does approximate nearest neighbor dudes search by projecting the data.
 */
public class ProjectionSearch extends UpdatableSearcher implements Iterable<Vector> {

  /**
   * A lists of tree sets containing the scalar projections of each vector.
   * The elements in a TreeMultiset are WeightedThing<Integer>, where the weight is the scalar
   * projection of the vector at the index pointed to by the Integer from the referenceVectors list
   * on the basis vector whose index is the same as the index of the TreeSet in the List.
   */
  private List<TreeMultiset<WeightedThing<Vector>>> scalarProjections;

  /**
   * The list of random normalized projection vectors forming a basis.
   * The TreeSet of scalar projections at index i in scalarProjections corresponds to the vector
   * at index i from basisVectors.
   */
  private Matrix basisMatrix;

  /**
   * The number of elements to consider on both sides in the ball around the vector found by the
   * search in a TreeSet from scalarProjections.
   */
  private int searchSize;

  private int numProjections;
  private boolean initialized = false;

  private void initialize(int numDimensions) {
    if (initialized)
      return;
    initialized = true;
    basisMatrix = RandomProjector.generateBasisNormal(numProjections, numDimensions);
    scalarProjections = Lists.newArrayList();
    for (int i = 0; i < numProjections; ++i) {
      scalarProjections.add(TreeMultiset.<WeightedThing<Vector>>create());
    }
  }

  public ProjectionSearch(DistanceMeasure distanceMeasure, int numProjections,  int searchSize) {
    super(distanceMeasure);
    Preconditions.checkArgument(numProjections > 0 && numProjections < 100,
        "Unreasonable value for number of projections");

    this.searchSize = searchSize;
    this.numProjections = numProjections;
  }

  /**
   * Adds a WeightedVector into the set of projections for later searching.
   * @param v  The WeightedVector to add.
   */
  @Override
  public void add(Vector v) {
    initialize(v.size());
    Vector projection = basisMatrix.times(v);
    // Add the the new vector and the projected distance to each set separately.
    int i = 0;
    for (TreeMultiset<WeightedThing<Vector>> s : scalarProjections) {
      s.add(new WeightedThing<Vector>(v, projection.get(i++)));
    }
    int numVectors = scalarProjections.get(0).size();
    for (TreeMultiset<WeightedThing<Vector>> s : scalarProjections) {
      Preconditions.checkArgument(s.size() == numVectors, "Number of vectors in projection sets " +
          "differ");
      double firstWeight = s.firstEntry().getElement().getWeight();
      for (WeightedThing<Vector> w : s) {
        Preconditions.checkArgument(firstWeight <= w.getWeight(), "Weights not in non-decreasing " +
            "order");
        firstWeight = w.getWeight();
      }
    }
  }

  /**
   * Returns the number of scalarProjections that we can search
   * @return  The number of scalarProjections added to the search so far.
   */
  public int size() {
    if (scalarProjections == null)
      return 0;
    return scalarProjections.get(0).size();
  }

  /**
   * Searches for the query vector returning the closest limit referenceVectors.
   *
   * @param query the vector to search for.
   * @param limit the number of results to return.
   * @return a list of Vectors wrapped in WeightedThings where the "thing"'s weight is the
   * distance.
   */
  public List<WeightedThing<Vector>> search(final Vector query, int limit) {
    HashSet<Vector> candidates = Sets.newHashSet();

    Iterator<? extends Vector> projections = basisMatrix.iterator();
    for (TreeMultiset<WeightedThing<Vector>> v : scalarProjections) {
      Vector basisVector = projections.next();
      WeightedThing<Vector> projectedQuery = new WeightedThing<Vector>(query,
          query.dot(basisVector));
      for (WeightedThing<Vector> candidate : Iterables.concat(
          Iterables.limit(v.tailMultiset(projectedQuery, BoundType.CLOSED), searchSize),
          Iterables.limit(v.headMultiset(projectedQuery, BoundType.OPEN).descendingMultiset(), searchSize))) {
        candidates.add(candidate.getValue());
      }
    }

    // If searchSize * scalarProjections.size() is small enough not to cause much memory pressure,
    // this is probably just as fast as a priority queue here.
    List<WeightedThing<Vector>> top = Lists.newArrayList();
    for (Vector candidate : candidates) {
      top.add(new WeightedThing<Vector>(candidate, distanceMeasure.distance(query, candidate)));
    }
    Collections.sort(top);
    return top.subList(0, Math.min(limit, top.size()));
  }

  @Override
  public Iterator<Vector> iterator() {
    return new AbstractIterator<Vector>() {
      private Iterator<WeightedThing<Vector>> projected = scalarProjections.get(0).iterator();
      @Override
      protected Vector computeNext() {
        if (!projected.hasNext()) {
          return endOfData();
        }
        return projected.next().getValue();
      }
    };
  }

  public boolean remove(Vector vector, double epsilon) {
    List<WeightedThing<Vector>> x = search(vector, 1);
    if (x.get(0).getWeight() < epsilon) {
      Iterator<? extends Vector> basisVectors = basisMatrix.iterator();
      for (TreeMultiset<WeightedThing<Vector>> projection : scalarProjections) {
        if (!projection.remove(new WeightedThing<Vector>(vector, vector.dot(basisVectors.next())))) {
          throw new RuntimeException("Internal inconsistency in ProjectionSearch");
        }
      }
      return true;
    } else {
      System.out.printf("%s to %s : %f ", vector, x.get(0).getValue(), x.get(0).getWeight());
      return false;
    }
  }

  @Override
  public void clear() {
    if (scalarProjections == null) {
      return;
    }
    for (TreeMultiset<WeightedThing<Vector>> set : scalarProjections) {
      set.clear();
    }
  }
}
