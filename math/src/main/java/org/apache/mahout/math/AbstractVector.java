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

package org.apache.mahout.math;

import java.util.Iterator;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.set.OpenIntHashSet;

import com.google.common.base.Preconditions;
/** Implementations of generic capabilities like sum of elements and dot products */
public abstract class AbstractVector implements Vector, LengthCachingVector {

  private static final double LOG2 = Math.log(2.0);

  private int size;
  protected double lengthSquared = -1.0;

  protected AbstractVector(int size) {
    this.size = size;
  }

  // TODO: Investigate the kinds of functions used here and possible optimizations.
  @Override
  public double aggregate(DoubleDoubleFunction aggregator, DoubleFunction map) {
    if (size < 1) {
      throw new IllegalArgumentException("Cannot aggregate empty vector");
    }
    double result = map.apply(getQuick(0));
    for (int i = 1; i < size; i++) {
      result = aggregator.apply(result, map.apply(getQuick(i)));
    }
    return result;
  }

  protected double aggregateLikeDotProductIterateOne(double result,
                                                     Iterator<Element> thisIterator, Vector other,
                                                     DoubleDoubleFunction aggregator, DoubleDoubleFunction combiner,
                                                     boolean swap) {
    Element thisElement;
    Element thatElement;
    while (thisIterator.hasNext()) {
      thisElement = thisIterator.next();
      thatElement = other.getElement(thisElement.index());
      if (thatElement.get() != 0) {
        result = aggregator.apply(result,
            swap ? combiner.apply(thatElement.get(), thisElement.get()) :
                combiner.apply(thisElement.get(), thatElement.get()));
      }
    }
    return result;
  }

  protected double aggregateLikeDotProductIterateBoth(double result,
                                                      Iterator<Element> thisIterator, Iterator<Element> thatIterator,
                                                      DoubleDoubleFunction aggregator, DoubleDoubleFunction combiner) {
    Element thisElement = null;
    Element thatElement = null;
    boolean advanceThis = true;
    boolean advanceThat = true;
    while (thisIterator.hasNext() && thatIterator.hasNext()) {
      if (advanceThis) {
        thisElement = thisIterator.next();
      }
      if (advanceThat) {
        thatElement = thatIterator.next();
      }
      if (thisElement.index() == thatElement.index()) {
        result = aggregator.apply(result, combiner.apply(thisElement.get(), thatElement.get()));
        advanceThis = true;
        advanceThat = true;
      } else {
        if (thisElement.index() < thatElement.index()) { // f(x, 0) = 0
          advanceThis = true;
          advanceThat = false;
        } else { // f(0, y) = 0
          advanceThis = false;
          advanceThat = true;
        }
      }
    }
    return result;
  }

  public double aggregateLikeDotProduct(Vector other, DoubleDoubleFunction aggregator, DoubleDoubleFunction combiner) {
    Preconditions.checkArgument(size == other.size(), "Vector sizes differ");
    Preconditions.checkArgument(size > 0, "Cannot aggregate empty vectors");

    Iterator<Element> thisIterator = iterateNonZero();
    Iterator<Element> thatIterator = other.iterateNonZero();
    if (!thisIterator.hasNext() || !thatIterator.hasNext()) {
      return 0;
    }
    Element thisElement = thisIterator.next();
    Element thatElement = thatIterator.next();
    double result;
    if (thisElement.index() == thatElement.index()) {
      result = combiner.apply(thisElement.get(), thatElement.get());
    } else {
      result = 0;
      thisIterator = iterateNonZero();
      thatIterator = other.iterateNonZero();
    }

    int numNondefaultThis = this.getNumNondefaultElements();
    int numNondefaultThat = other.getNumNondefaultElements();
    double bothCost = numNondefaultThis + numNondefaultThat;
    double oneCostThis = other.isRandomAccess() ?
        numNondefaultThis : (numNondefaultThis * Functions.LOG2.apply(numNondefaultThat));
    double oneCostThat = isRandomAccess() ?
        numNondefaultThat : (numNondefaultThat * Functions.LOG2.apply(numNondefaultThis));

    if (oneCostThis <= oneCostThat && oneCostThis <= bothCost) {
      return aggregateLikeDotProductIterateOne(result, thisIterator, other, aggregator, combiner, false);
    }
    if (oneCostThat < oneCostThis && oneCostThat <= bothCost) {
      return aggregateLikeDotProductIterateOne(result, thatIterator, this, aggregator, combiner, true);
    }
    return aggregateLikeDotProductIterateBoth(result, thisIterator, thatIterator, aggregator, combiner);
  }

  @Override
  public double aggregate(Vector other, DoubleDoubleFunction aggregator, DoubleDoubleFunction combiner) {
    Preconditions.checkArgument(size == other.size(), "Vector sizes differ");
    Preconditions.checkArgument(size > 0, "Cannot aggregate empty vectors");

    if ((isSequentialAccess() && other.isSequentialAccess()) || aggregator.isCommutative()) {
      if (aggregator.isLikeRightPlus() && combiner.isLikeRightMult()) {
        return aggregateLikeDotProduct(other, aggregator, combiner);
      }

      Iterator<Element> thisIterator = this.iterator();
      Iterator<Element> thatIterator = other.iterator();
      Element thisElement = thisIterator.next();
      Element thatElement = thatIterator.next();
      double result = combiner.apply(thisElement.get(), thatElement.get());
      while (thisIterator.hasNext() && thatIterator.hasNext()) {
        thisElement = thisIterator.next();
        thatElement = thatIterator.next();
        result = aggregator.apply(result, combiner.apply(thisElement.get(), thatElement.get()));
      }
      return result;
    } else {
      double result = combiner.apply(getQuick(0), other.getQuick(0));
      for (int i = 1; i < size(); ++i) {
        result = aggregator.apply(result, combiner.apply(getQuick(i), other.getQuick(i)));
      }
      return result;
    }
  }

  /**
   * Subclasses must override to return an appropriately sparse or dense result
   *
   * @param rows    the row cardinality
   * @param columns the column cardinality
   * @return a Matrix
   */
  protected abstract Matrix matrixLike(int rows, int columns);

  @Override
  public Vector viewPart(int offset, int length) {
    if (offset < 0) {
      throw new IndexException(offset, size);
    }
    if (offset + length > size) {
      throw new IndexException(offset + length, size);
    }
    return new VectorView(this, offset, length);
  }

  @Override
  public Vector clone() {
    try {
      AbstractVector r = (AbstractVector) super.clone();
      r.size = size;
      r.lengthSquared = lengthSquared;
      return r;
    } catch (CloneNotSupportedException e) {
      throw new IllegalStateException("Can't happen");
    }
  }

  @Override
  public Vector divide(double x) {
    if (x == 1.0) {
      return clone();
    }
    Vector result = createOptimizedCopy();
    Iterator<Element> iter = result.iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      element.set(element.get() / x);
    }
    return result;
  }

  @Override
  public double dot(Vector x) {
    if (size != x.size()) {
      throw new CardinalityException(size, x.size());
    }
    if (this == x) {
      return getLengthSquared();
    }

    // Crude rule of thumb: when a sequential-access vector, with O(log n) lookups, has about
    // 2^n elements, its lookups take longer than a dense / random access vector (with O(1) lookups) by
    // about a factor of (0.71n - 12.3). This holds pretty well from n=19 up to at least n=23 according to my tests;
    // below that lookups are so fast that this difference is near zero.

    int thisNumNonDefault = getNumNondefaultElements();
    int thatNumNonDefault = x.getNumNondefaultElements();
    // Default: dot from smaller vector to larger vector
    boolean reverseDot = thatNumNonDefault < thisNumNonDefault;

    // But, see if we should override that -- is exactly one of them sequential access and so slower to lookup in?
    if (isSequentialAccess() != x.isSequentialAccess()) {
      double log2ThisSize = Math.log(thisNumNonDefault) / LOG2;
      double log2ThatSize = Math.log(thatNumNonDefault) / LOG2;
      // Only override when the O(log n) factor seems big enough to care about:
      if (log2ThisSize >= 19.0 && log2ThatSize >= 19.0) {
        double dotCost = thisNumNonDefault;
        if (x.isSequentialAccess()) {
          dotCost *= 0.71 * log2ThatSize - 12.3;
        }
        double reverseDotCost = thatNumNonDefault;
        if (isSequentialAccess()) {
          reverseDotCost *= 0.71 * log2ThisSize - 12.3;
        }
        reverseDot = reverseDotCost < dotCost;
      }
    } else if (this.isSequentialAccess()  && !this.isDense() && x.isSequentialAccess() && !x.isDense()) {
      Element thisElement = null;
      Element thatElement = null;
      boolean advanceThis = true;
      boolean advanceThat = true;

      Iterator<Element> thisNonZero = this.iterateNonZero();
      Iterator<Element> thatNonZero = x.iterateNonZero();

      double result = 0.0;
      while (true) {
        if (advanceThis) {
          if (!thisNonZero.hasNext()) {
            break;
          }
          thisElement = thisNonZero.next();
        }
        if (advanceThat) {
          if (!thatNonZero.hasNext()) {
            break;
          }
          thatElement = thatNonZero.next();
        }
        if (thisElement.index() == thatElement.index()) {
          result += thisElement.get() * thatElement.get();
          advanceThis = true;
          advanceThat = true;
        } else if (thisElement.index() < thatElement.index()) {
          advanceThis = true;
          advanceThat = false;
        } else {
          advanceThis = false;
          advanceThat = true;
        }
      }
      return result;
    }


    if (reverseDot) {
      return x.dot(this);
    }

    double result = 0.0;
    Iterator<Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      result += element.get() * x.getQuick(element.index());
    }
    return result;
  }

  // TODO: should be in terms of aggregate
  protected double dotSelf() {
    double result = 0.0;
    Iterator<Element> i = iterateNonZero();
    while (i.hasNext()) {
      double value = i.next().get();
      result += value * value;
    }
    return result;
  }

  @Override
  public double get(int index) {
    if (index < 0 || index >= size) {
      throw new IndexException(index, size);
    }
    return getQuick(index);
  }

  @Override
  public Element getElement(int index) {
    return new LocalElement(index);
  }

  // TODO: should be in terms of assign
  @Override
  public Vector normalize() {
    return divide(Math.sqrt(getLengthSquared()));
  }

  @Override
  public Vector normalize(double power) {
    return divide(norm(power));
  }

  @Override
  public Vector logNormalize() {
    return logNormalize(2.0, Math.sqrt(getLengthSquared()));
  }

  @Override
  public Vector logNormalize(double power) {
    return logNormalize(power, norm(power));
  }

  public Vector logNormalize(double power, double normLength) {
    // we can special case certain powers
    if (Double.isInfinite(power) || power <= 1.0) {
      throw new IllegalArgumentException("Power must be > 1 and < infinity");
    } else {
      double denominator = normLength * Math.log(power);
      Vector result = createOptimizedCopy();
      Iterator<Element> iter = result.iterateNonZero();
      while (iter.hasNext()) {
        Element element = iter.next();
        element.set(Math.log1p(element.get()) / denominator);
      }
      return result;
    }
  }

  // p-norm, where p = power
  // TODO: express as an aggregate
  @Override
  public double norm(double power) {
    if (power < 0.0) {
      throw new IllegalArgumentException("Power must be >= 0");
    }
    // we can special case certain powers
    if (Double.isInfinite(power)) {
      double val = 0.0;
      Iterator<Element> iter = this.iterateNonZero();
      while (iter.hasNext()) {
        val = Math.max(val, Math.abs(iter.next().get()));
      }
      return val;
    } else if (power == 2.0) {
      return Math.sqrt(getLengthSquared());
    } else if (power == 1.0) {
      double val = 0.0;
      Iterator<Element> iter = this.iterateNonZero();
      while (iter.hasNext()) {
        val += Math.abs(iter.next().get());
      }
      return val;
    } else if (power == 0.0) {
      // this is the number of non-zero elements
      double val = 0.0;
      Iterator<Element> iter = this.iterateNonZero();
      while (iter.hasNext()) {
        val += iter.next().get() == 0 ? 0 : 1;
      }
      return val;
    } else {
      double val = 0.0;
      Iterator<Element> iter = this.iterateNonZero();
      while (iter.hasNext()) {
        Element element = iter.next();
        val += Math.pow(element.get(), power);
      }
      return Math.pow(val, 1.0 / power);
    }
  }

  @Override
  public double getLengthSquared() {
    if (lengthSquared >= 0.0) {
      return lengthSquared;
    }
    return lengthSquared = dotSelf();
  }

  @Override
  public void invalidateCachedLength() {
    lengthSquared = -1;
  }

  @Override
  public double getDistanceSquared(Vector v) {
    if (size != v.size()) {
      throw new CardinalityException(size, v.size());
    }
    // if this and v has a cached lengthSquared, dot product is quickest way to compute this.
    double d1;
    double d2;
    double dot;
    if (lengthSquared >= 0) {
      // our length squared is cached.  use it
      // the max is (slight) antidote to round-off errors
      d1 = lengthSquared;
      d2 = v.getLengthSquared();
      dot = this.dot(v);
    } else {
      // our length is not cached... compute it and the dot product in one pass for speed
      d1 = 0;
      d2 = v.getLengthSquared();
      dot = 0;
      final Iterator<Element> i = iterateNonZero();
      while (i.hasNext()) {
        Element e = i.next();
        double value = e.get();
        d1 += value * value;
        dot += value * v.getQuick(e.index());
      }
      lengthSquared = d1;
      // again, round-off errors may be present
    }

    double r = d1 + d2 - 2 * dot;
    if (r > 1.0e-3 * (d1 + d2)) {
      return Math.max(0, r);
    } else {
      if (this.isSequentialAccess()) {
        if (v.isSequentialAccess()) {
          return mergeDiff(this, v);
        } else {
          return randomScanDiff(this, v);
        }
      } else {
        return randomScanDiff(v, this);
      }
    }
  }

  /**
   * Computes the squared difference of two vectors where iterateNonZero
   * is efficient for each vector, but where the order of iteration is not
   * known.  This forces us to access most elements of v2 via get(), which
   * would be very inefficient for some kinds of vectors.
   *
   * Note that this static method is exposed at a package level for testing purposes only.
   * @param v1  The vector that we access only via iterateNonZero
   * @param v2  The vector that we access via iterateNonZero and via Element.get()
   * @return The squared difference between v1 and v2.
   */
  static double randomScanDiff(Vector v1, Vector v2) {
    // keeps a list of elements we visited by iterating over v1.  This should be
    // almost all of the elements of v2 because we only call this method if the
    // difference is small.
    OpenIntHashSet visited = new OpenIntHashSet();

    double r = 0;

    // walk through non-zeros of v1
    Iterator<Element> i = v1.iterateNonZero();
    while (i.hasNext()) {
      Element e1 = i.next();
      visited.add(e1.index());
      double x = e1.get() - v2.get(e1.index());
      r += x * x;
    }

    // now walk through neglected elements of v2
    i = v2.iterateNonZero();
    while (i.hasNext()) {
      Element e2 = i.next();
      if (!visited.contains(e2.index())) {
        // if not visited already then v1's value here would be zero.
        double x = e2.get();
        r += x * x;
      }
    }

    return r;
  }

  /**
   * Computes the squared difference of two vectors where iterateNonZero returns
   * elements in index order for both vectors.  This allows a merge to be used to
   * compute the difference.  A merge allows a single sequential pass over each
   * vector and should be faster than any alternative.
   *
   * Note that this static method is exposed at a package level for testing purposes only.
   * @param v1  The first vector.
   * @param v2  The second vector.
   * @return The squared difference between the two vectors.
   */
  static double mergeDiff(Vector v1, Vector v2) {
    Iterator<Element> i1 = v1.iterateNonZero();
    Iterator<Element> i2 = v2.iterateNonZero();

    // v1 is empty?
    if (!i1.hasNext()) {
      return v2.getLengthSquared();
    }

    // v2 is empty?
    if (!i2.hasNext()) {
      return v1.getLengthSquared();
    }

    Element e1 = i1.next();
    Element e2 = i2.next();

    double r = 0;
    while (e1 != null && e2 != null) {
      // eat elements of v1 that precede all in v2
      while (e1 != null && e1.index() < e2.index()) {
        double x = e1.get();
        r += x * x;

        if (i1.hasNext()) {
          e1 = i1.next();
        } else {
          e1 = null;
        }
      }

      // at this point we have three possibilities, e1 == null or e1 matches e2 or
      // e2 precedes e1.  Here we handle the e2 < e1 case
      while (e2 != null && (e1 == null || e2.index() < e1.index())) {
        double x = e2.get();
        r += x * x;

        if (i2.hasNext()) {
          e2 = i2.next();
        } else {
          e2 = null;
        }
      }

      // and now we handle the e1 == e2 case.  For convenience, we
      // grab as many of these as possible.  Given that we are called here
      // only when v1 and v2 are nearly equal, this loop should dominate
      while (e1 != null && e2 != null && e1.index() == e2.index()) {
        double x = e1.get() - e2.get();
        r += x * x;

        if (i1.hasNext()) {
          e1 = i1.next();
        } else {
          e1 = null;
        }

        if (i2.hasNext()) {
          e2 = i2.next();
        } else {
          e2 = null;
        }
      }
    }

    // one of i1 or i2 is exhausted here, but the other may not be
    while (e1 != null) {
      double x = e1.get();
      r += x * x;

      if (i1.hasNext()) {
        e1 = i1.next();
      } else {
        e1 = null;
      }
    }

    while (e2 != null) {
      double x = e2.get();
      r += x * x;

      if (i2.hasNext()) {
        e2 = i2.next();
      } else {
        e2 = null;
      }
    }
    // both v1 and v2 have been completely processed
    return r;
  }

  // TODO: express as aggregate
  @Override
  public double maxValue() {
    double result = Double.NEGATIVE_INFINITY;
    int nonZeroElements = 0;
    Iterator<Element> iter = this.iterateNonZero();
    while (iter.hasNext()) {
      nonZeroElements++;
      Element element = iter.next();
      result = Math.max(result, element.get());
    }
    if (nonZeroElements < size) {
      return Math.max(result, 0.0);
    }
    return result;
  }

  // TODO: express as aggregate
  @Override
  public int maxValueIndex() {
    int result = -1;
    double max = Double.NEGATIVE_INFINITY;
    int nonZeroElements = 0;
    Iterator<Element> iter = this.iterateNonZero();
    while (iter.hasNext()) {
      nonZeroElements++;
      Element element = iter.next();
      double tmp = element.get();
      if (tmp > max) {
        max = tmp;
        result = element.index();
      }
    }
    // if the maxElement is negative and the vector is sparse then any
    // unfilled element(0.0) could be the maxValue hence we need to
    // find one of those elements
    if (nonZeroElements < size && max < 0.0) {
      for (Element element : this) {
        if (element.get() == 0.0) {
          return element.index();
        }
      }
    }
    return result;
  }

  // TODO: express as aggregate
  @Override
  public double minValue() {
    double result = Double.POSITIVE_INFINITY;
    int nonZeroElements = 0;
    Iterator<Element> iter = this.iterateNonZero();
    while (iter.hasNext()) {
      nonZeroElements++;
      Element element = iter.next();
      result = Math.min(result, element.get());
    }
    if (nonZeroElements < size) {
      return Math.min(result, 0.0);
    }
    return result;
  }

  // TODO: express as aggregate
  @Override
  public int minValueIndex() {
    int result = -1;
    double min = Double.POSITIVE_INFINITY;
    int nonZeroElements = 0;
    Iterator<Element> iter = this.iterateNonZero();
    while (iter.hasNext()) {
      nonZeroElements++;
      Element element = iter.next();
      double tmp = element.get();
      if (tmp < min) {
        min = tmp;
        result = element.index();
      }
    }
    // if the maxElement is positive and the vector is sparse then any
    // unfilled element(0.0) could be the maxValue hence we need to
    // find one of those elements
    if (nonZeroElements < size && min > 0.0) {
      for (Element element : this) {
        if (element.get() == 0.0) {
          return element.index();
        }
      }
    }
    return result;
  }

  // TODO: express as aggregate
  @Override
  public Vector plus(double x) {
    Vector result = createOptimizedCopy();
    if (x == 0.0) {
      return result;
    }
    int size = result.size();
    for (int i = 0; i < size; i++) {
      result.incrementQuick(i, x);
    }
    return result;
  }

  // TODO: express as aggregate
  @Override
  public Vector plus(Vector x) {
    if (size != x.size()) {
      throw new CardinalityException(size, x.size());
    }

    Vector result = createOptimizedCopy();

    Iterator<Element> iter = x.iterateNonZero();
    while (iter.hasNext()) {
      Element e = iter.next();
      int index = e.index();
      result.incrementQuick(index, e.get());
    }
    return result;
  }

  @Override
  public Vector minus(Vector that) {
    if (size != that.size()) {
      throw new CardinalityException(size, that.size());
    }

    Vector result = createOptimizedCopy();

    Iterator<Element> iter = that.iterateNonZero();
    while (iter.hasNext()) {
      Element thatElement = iter.next();
      int index = thatElement.index();
      result.incrementQuick(index, -thatElement.get());
    }
    return result;
  }

  @Override
  public void set(int index, double value) {
    if (index < 0 || index >= size) {
      throw new IndexException(index, size);
    }
    setQuick(index, value);
  }

  // TODO: express as aggregate
  @Override
  public void incrementQuick(int index, double increment) {
    setQuick(index, getQuick(index) + increment);
  }

  @Override
  public Vector times(double x) {
    if (x == 0.0) {
      return like();
    }

    Vector result = createOptimizedCopy();

    if (x == 1.0) {
      return result;
    }

    Iterator<Element> iter = result.iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      element.set(element.get() * x);
    }

    return result;
  }

  /**
   * Copy the current vector in the most optimum fashion. Used by immutable methods like plus(), minus().
   * Use this instead of vector.like().assign(vector). Sub-class can choose to override this method.
   *
   * @return a copy of the current vector.
   */
  protected Vector createOptimizedCopy() {
    return createOptimizedCopy(this);
  }

  private Vector createOptimizedCopy(Vector v) {
    Vector result;
    if (isDense()) {
      result = v.like().assign(v);
    } else {
      result = v.clone();
    }
    return result;
  }

  // TODO: express as assign
  @Override
  public Vector times(Vector x) {
    if (size != x.size()) {
      throw new CardinalityException(size, x.size());
    }

    Vector to = this;
    Vector from = x;
    // Clone and edit to the sparse one; if both are sparse, edit the more sparse one (more zeroes)
    if (isDense() || !x.isDense() && getNumNondefaultElements() > x.getNumNondefaultElements()) {
      to = x;
      from = this;
    }

    Vector result = createOptimizedCopy(to);
    Iterator<Element> iter = result.iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      element.set(element.get() * from.getQuick(element.index()));
    }

    return result;
  }

  @Override
  public double zSum() {
    double result = 0.0;
    Iterator<Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      result += iter.next().get();
    }

    return result;
  }

  @Override
  public int getNumNonZeroElements() {
    int count = 0;
    Iterator<Element> it = iterateNonZero();
    while (it.hasNext()) {
      if (it.next().get() != 0.0) {
        count++;
      }
    }
    return count;
  }

  // TODO: use iterators
  @Override
  public Vector assign(double value) {
    for (int i = 0; i < size; i++) {
      setQuick(i, value);
    }
    return this;
  }

  // TODO: use iterators
  @Override
  public Vector assign(double[] values) {
    if (size != values.length) {
      throw new CardinalityException(size, values.length);
    }
    for (int i = 0; i < size; i++) {
      setQuick(i, values[i]);
    }
    return this;
  }

  // TODO: use iterators
  @Override
  public Vector assign(Vector other) {
    if (size != other.size()) {
      throw new CardinalityException(size, other.size());
    }
    for (int i = 0; i < size; i++) {
      setQuick(i, other.getQuick(i));
    }
    return this;
  }

  @Override
  public Vector assign(DoubleDoubleFunction f, double y) {
    Iterator<Element> it = f.apply(0, y) == 0 ? iterateNonZero() : iterator();
    while (it.hasNext()) {
      Element e = it.next();
      e.set(f.apply(e.get(), y));
    }
    return this;
  }

  @Override
  public Vector assign(DoubleFunction function) {
    Iterator<Element> it = function.apply(0) == 0 ? iterateNonZero() : iterator();
    while (it.hasNext()) {
      Element e = it.next();
      e.set(function.apply(e.get()));
    }
    return this;
  }

  /*
  @Override
  public Vector assign(Vector other, DoubleDoubleFunction function) {
    if (size != other.size()) {
      throw new CardinalityException(size, other.size());
    }

    // special case: we only need to iterate over the non-zero elements of the vector to add
    if (Functions.PLUS.equals(function) || Functions.PLUS_ABS.equals(function)) {
      Iterator<Vector.Element> nonZeroElements = other.iterateNonZero();
      while (nonZeroElements.hasNext()) {
        Vector.Element e = nonZeroElements.next();
        setQuick(e.index(), function.apply(getQuick(e.index()), e.get()));
      }
    } else {
      for (int i = 0; i < size; i++) {
        setQuick(i, function.apply(getQuick(i), other.getQuick(i)));
      }
    }
    return this;
  }
  */

  /**
   * Assigns the current vector, x the value of f(x, y) where f is applied to every component of x and y sequentially.
   * xi = f(xi, yi).
   * Let:
   * - d = cardinality of x and y (they must be equal for the assignment to be possible);
   * - nx = number of nonzero elements in x;
   * - ny = number of nonzero elements in y;
   *
   * If the function is densifying (f(0, 0) != 0), the resulting vector will be dense and the complexity
   * of this function is O(d).
   * Otherwise, the worst-case complexity if O(nx + ny).
   *
   * Note, that for two classes of functions we can do better:
   * a. if f(x, 0) = x (function instanceof LikeRightPlus), the zeros in y can't affect the values of x, so it's
   * enough to iterate through the nonzero values of y to compute the result. For each nonzero yi, we need to find
   * the corresponding xi and update it accordingly.
   * b. if f(0, y) = 0 (function instanceof LikeLeftMult), the zeros in x are not affected by any value, so it's
   * enough to iterate through the nonzero values of x to compute the result. For each nonzero xi, we need to find
   * the corresponding yi and update xi accordingly.
   * Finding a random element in a vector given its index however is O(log n) for sequential access sparse vectors
   * (because of binary search).
   * Therefore, in case a. if x is sequential, the complexity is O(ny * log nx) otherwise it's O(ny + nx).
   * Similarly, in case a. if x is sequential, the complexity is O(nx * log ny) otherwise it's O(nx + ny).
   *
   * Practically, we use a. or b. even for sequential vectors when m * log n < m + n that is m < n / (log n - 1),
   * where n is the number of nonzero elements in the sequential vector and m is the number of elements in the other
   * vector.
   *
   * @param other    a Vector containing the second arguments to the function (y).
   * @param function a DoubleDoubleFunction to apply (f).
   * @return this vector, after being updated.
   */
  @Override
  public Vector assign(Vector other, DoubleDoubleFunction function) {
    if (size != other.size()) {
      throw new CardinalityException(size, other.size());
    }

    boolean isDensifying = function.apply(0, 0) != 0;
    if (isDensifying) {
      // Sanity checks that the function don't claim to implement the special interfaces.
      if (function.isLikeRightPlus()) {
        throw new IllegalArgumentException("Invalid function definition. f(0, 0) != 0 but it claims that f(x, 0) = x");
      }
      if (function.isLikeLeftMult()) {
        throw new IllegalArgumentException("Invalid function definition. f(0, 0) != 0 but it claims that f(0, y) = 0");
      }
    }

    Iterator<Element> thisIterator;
    Iterator<Element> thatIterator;
    Element thisElement;
    Element thatElement;
    // The resulting vector will be dense so we'll just iterate through all the elements.
    if (isDensifying || !isSequentialAccess() || !other.isSequentialAccess()) {
      if (isSequentialAccess() && other.isSequentialAccess()) {
        thisIterator = this.iterator();
        thatIterator = other.iterator();
        while (thisIterator.hasNext() && thatIterator.hasNext()) {
          thisElement = thisIterator.next();
          thisElement.set(function.apply(thisElement.get(), thatIterator.next().get()));
        }
      } else {
        for (int i = 0; i < size; ++i) {
          setQuick(i, function.apply(getQuick(i), other.getQuick(i)));
        }
      }
    } else {
      // We can get away with just iterating through the non-zero elements in both vectors.
      // If however our function is special (xZeroX or zeroYZero is true) we can just iterate through one of the
      // vectors.
      double nx = getNumNondefaultElements();
      double ny = other.getNumNondefaultElements();
      if (function.isLikeRightPlus() && (isRandomAccess()
          || (!isRandomAccess() && ny < (nx / (Functions.LOG2.apply(nx) - 1))))) {
        // When f(x, 0) = x, the 0s in "other" will not affect "this" in any way. So, we can just iterate though
        // the non-zero values in "other" and apply the function where we need to.
        thatIterator = other.iterateNonZero();
        OrderedIntDoubleMapping thisUpdates = new OrderedIntDoubleMapping();
        while (thatIterator.hasNext()) {
          thatElement = thatIterator.next();
          thisElement = this.getElement(thatElement.index());
          if (thisElement.get() == 0) {
            thisUpdates.set(thatElement.index(), function.apply(0, thatElement.get()));
          } else {
            thisElement.set(function.apply(thisElement.get(), thatElement.get()));
          }
        }
        mergeUpdates(thisUpdates);
      } else if (function.isLikeLeftMult() && (other.isRandomAccess()
          || (!other.isRandomAccess() && nx < (ny / Functions.LOG2.apply(ny) - 1)))) {
        // When f(0, y) = 0, the 0s in "this" will not be affected "this" in any way. So, we can just iterate though
        // the non-zero values in "other" and apply the function where we need to.
        thisIterator = iterateNonZero();
        while (thisIterator.hasNext()) {
          thisElement = thisIterator.next();
          thatElement = other.getElement(thisElement.index());
          thisElement.set(function.apply(thisElement.get(), thatElement.get()));
        }
      } else {
        thisIterator = this.iterateNonZero();
        thatIterator = other.iterateNonZero();
        thisElement = thatElement = null;
        boolean advanceThis = true;
        boolean advanceThat = true;
        OrderedIntDoubleMapping thisUpdates = new OrderedIntDoubleMapping();

        while (thisIterator.hasNext() && thatIterator.hasNext()) {
          if (advanceThis) {
            thisElement = thisIterator.next();
          }
          if (advanceThat) {
            thatElement = thatIterator.next();
          }
          if (thisElement.index() == thatElement.index()) {
            thisElement.set(function.apply(thisElement.get(), thatElement.get()));
            advanceThis = true;
            advanceThat = true;
          } else {
            if (thisElement.index() < thatElement.index()) { // f(x, 0)
              thisElement.set(function.apply(thisElement.get(), 0));
              advanceThis = true;
              advanceThat = false;
            } else {
              double result = function.apply(0, thatElement.get());
              if (result != 0) { // f(0, y) != 0
                thisUpdates.set(thatElement.index(), result);
              }
              advanceThis = false;
              advanceThat = true;
            }
          }
        }

        while (thisIterator.hasNext()) {
          thisElement = thisIterator.next();
          thisElement.set(function.apply(thisElement.get(), 0));
        }
        while (thatIterator.hasNext()) {
          thatElement = thatIterator.next();
          double result = function.apply(0, thatElement.get());
          if (result != 0) {
            thisUpdates.set(thatElement.index(), result);
          }
        }
        mergeUpdates(thisUpdates);
      }
    }
    return this;
  }

  /**
   * Used internally by assign() to update multiple indices and values at once.
   * Only really useful for sparse vectors (especially SequentialAccessSparseVector).
   *
   * If someone ever adds a new type of sparse vectors, this method must merge (index, value) pairs into the vector.
   *
   * @param updates a mapping of indices to values to merge in the vector.
   */
  public abstract void mergeUpdates(OrderedIntDoubleMapping updates);

  // TODO: looks fishy, why the getQuick()?
  @Override
  public Matrix cross(Vector other) {
    Matrix result = matrixLike(size, other.size());
    for (int row = 0; row < size; row++) {
      result.assignRow(row, other.times(getQuick(row)));
    }
    return result;
  }

  @Override
  public final int size() {
    return size;
  }

  @Override
  public String asFormatString() {
    return toString();
  }

  @Override
  public int hashCode() {
    int result = size;
    Iterator<Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      Element ele = iter.next();
      result += ele.index() * RandomUtils.hashDouble(ele.get());
    }
    return result;
  }

  /**
   * Determines whether this {@link Vector} represents the same logical vector as another
   * object. Two {@link Vector}s are equal (regardless of implementation) if the value at
   * each index is the same, and the cardinalities are the same.
   */
  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof Vector)) {
      return false;
    }
    Vector that = (Vector) o;
    if (size != that.size()) {
      return false;
    }
    for (int index = 0; index < size; index++) {
      if (getQuick(index) != that.getQuick(index)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public String toString() {
    return toString(null);
  }

  public String toString(String[] dictionary) {
    StringBuilder result = new StringBuilder();
    result.append('{');
    for (int index = 0; index < size; index++) {
      double value = getQuick(index);
      if (value != 0.0) {
        result.append(dictionary != null && dictionary.length > index ? dictionary[index] : index);
        result.append(':');
        result.append(value);
        result.append(',');
      }
    }
    if (result.length() > 1) {
      result.setCharAt(result.length() - 1, '}');
    } else {
      result.append('}');
    }
    return result.toString();
  }


  protected final class LocalElement implements Element {
    int index;

    LocalElement(int index) {
      this.index = index;
    }

    @Override
    public double get() {
      return getQuick(index);
    }

    @Override
    public int index() {
      return index;
    }

    @Override
    public void set(double value) {
      setQuick(index, value);
    }
  }
}
