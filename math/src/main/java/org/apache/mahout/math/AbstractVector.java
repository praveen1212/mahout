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

import com.google.common.base.Preconditions;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.jet.math.Constants;

import java.util.Iterator;
/** Implementations of generic capabilities like sum of elements and dot products */
public abstract class AbstractVector implements Vector, LengthCachingVector {

  private static final double LOG2 = Math.log(2.0);

  private int size;
  protected double lengthSquared = -1.0;

  protected AbstractVector(int size) {
    this.size = size;
  }

  /**
   * Let fa = aggregator and fm = map.
   * If fm(0) = 0 and fa(0, x) =
   *
   * @param aggregator used to combine the current value of the aggregation with the result of map.apply(nextValue)
   * @param map a function to apply to each element of the vector in turn before passing to the aggregator
   * @return
   */
  @Override
  public double aggregate(DoubleDoubleFunction aggregator, DoubleFunction map) {
    if (size < 1) {
      throw new IllegalArgumentException("Cannot aggregate empty vector");
    }

    boolean commutativeAndAssociative = aggregator.isCommutative() && aggregator.isAssociative();

    if (commutativeAndAssociative) {
      boolean hasZeros = size() - getNumNonZeroElements() > 0;
      if (hasZeros && Math.abs(map.apply(0.0) - 0.0) < Constants.EPSILON) {
        // There exists at least one zero, and fm(0) = 0. The results starts as 0.0.
        // This can be the first result in the aggregator (because it's associative and commutative).
        // The aggregator is applied as fa(result, fm(v)), but we know there must be a fa(0, fm(v)).
        // If f(0, y) = 0 (isLikeLeftMult), the 0 will cascade through the aggregation and the final result is 0.
        if (aggregator.isLikeLeftMult()){
          return 0.0;
        }
      }
    }

    double result;
    if (isSequentialAccess() || commutativeAndAssociative) {
      Iterator<Element> iterator;
      // If fm(0) = 0 and fa(x, 0) = x, we can skip all zero values.
      if (Math.abs(map.apply(0.0) - 0.0) < Constants.EPSILON && aggregator.isLikeRightPlus()) {
        iterator = iterateNonZero();
        if (!iterator.hasNext()) {
          return 0.0;
        }
      } else {
        iterator = iterator();
      }
      Element element = iterator.next();
      result = map.apply(element.get());
      while (iterator.hasNext()) {
        element = iterator.next();
        result = aggregator.apply(result, map.apply(element.get()));
      }
    } else {
      result = map.apply(getQuick(0));
      for (int i = 1; i < size; i++) {
        result = aggregator.apply(result, map.apply(getQuick(i)));
      }
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

    if ((Math.abs(combiner.apply(0.0, 0.0) - 0.0) < Constants.EPSILON)
        && ((isSequentialAccess() && other.isSequentialAccess())
            || (aggregator.isAssociative() && aggregator.isCommutative()))) {
      if (aggregator.isLikeRightPlus()) {
        return aggregateLikeDotProduct(other, aggregator, combiner);
      }  else {
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
      }
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
    return aggregate(x, Functions.PLUS, Functions.MULT);
  }

  protected double dotSelf() {
    return aggregate(Functions.PLUS, Functions.pow(2));
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
  @Override
  public double norm(double power) {
    if (power < 0.0) {
      throw new IllegalArgumentException("Power must be >= 0");
    }
    // we can special case certain powers
    if (Double.isInfinite(power)) {
      return aggregate(Functions.MAX, Functions.ABS);
    } else if (power == 2.0) {
      return Math.sqrt(getLengthSquared());
    } else if (power == 1.0) {
      return aggregate(Functions.PLUS, Functions.ABS);
    } else if (power == 0.0) {
      return getNumNonZeroElements();
    } else {
      return Math.pow(aggregate(Functions.PLUS, Functions.pow(power)), 1.0 / power);
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
    // return minus(v).aggregate(Functions.PLUS, Functions.pow(2));
    return aggregate(v, Functions.PLUS, Functions.MINUS_SQUARED);
  }

  @Override
  public double maxValue() {
    if (size == 0) {
      return Double.NEGATIVE_INFINITY;
    }
    return aggregate(Functions.MAX, Functions.IDENTITY);
  }

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

  @Override
  public double minValue() {
    if (size == 0) {
      return Double.POSITIVE_INFINITY;
    }
    return aggregate(Functions.MIN, Functions.IDENTITY);
  }

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

  @Override
  public Vector plus(double x) {
    Vector result = createOptimizedCopy();
    if (x == 0.0) {
      return result;
    }
    return result.assign(Functions.plus(x));
  }

  @Override
  public Vector plus(Vector that) {
    if (size != that.size()) {
      throw new CardinalityException(size, that.size());
    }
    return createOptimizedCopy().assign(that, Functions.PLUS);
  }

  @Override
  public Vector minus(Vector that) {
    if (size != that.size()) {
      throw new CardinalityException(size, that.size());
    }
    return createOptimizedCopy().assign(that, Functions.MINUS);
  }

  @Override
  public void set(int index, double value) {
    if (index < 0 || index >= size) {
      throw new IndexException(index, size);
    }
    setQuick(index, value);
  }

  @Override
  public void incrementQuick(int index, double increment) {
    setQuick(index, getQuick(index) + increment);
  }

  @Override
  public Vector times(double x) {
    if (x == 0.0) {
      return like();
    }
    return createOptimizedCopy().assign(Functions.mult(x));
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

  @Override
  public Vector times(Vector that) {
    if (size != that.size()) {
      throw new CardinalityException(size, that.size());
    }
    Vector to = this;
    Vector from = that;
    // Clone and edit to the sparse one; if both are sparse, edit the more sparse one (more zeroes)
    if (isDense() || !that.isDense() && getNumNondefaultElements() > that.getNumNondefaultElements()) {
      to = that;
      from = this;
    }
    return createOptimizedCopy(to).assign(from, Functions.MULT);
  }

  @Override
  public double zSum() {
    return aggregate(Functions.PLUS, Functions.IDENTITY);
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

  @Override
  public Vector assign(double value) {
    Iterator<Element> it;
    if (value == 0.0) {
      // Make all the non-zero values 0.
      it = iterateNonZero();
      while (it.hasNext()) {
        it.next().set(value);
      }
    } else {
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      // Update all the non-zero values and queue the updates for the zero vaues.
      // The vector will become dense.
      it = iterator();
      while (it.hasNext()) {
        Element element = it.next();
        if (element.get() == 0.0) {
          updates.set(element.index(), value);
        } else {
          element.set(value);
        }
      }
      mergeUpdates(updates);
    }
    invalidateCachedLength();
    return this;
  }

  @Override
  public Vector assign(double[] values) {
    if (size != values.length) {
      throw new CardinalityException(size, values.length);
    }
    OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
    Iterator<Element> it = iterator();
    while (it.hasNext()) {
      Element element = it.next();
      int index = element.index();
      if (element.get() == 0.0) {
        updates.set(index, values[index]);
      } else {
        element.set(values[index]);
      }
    }
    mergeUpdates(updates);
    invalidateCachedLength();
    return this;
  }

  @Override
  public Vector assign(Vector other) {
    return assign(other, Functions.SECOND);
  }

  @Override
  public Vector assign(DoubleDoubleFunction f, double y) {
    Iterator<Element> it = f.apply(0, y) == 0 ? iterateNonZero() : iterator();
    while (it.hasNext()) {
      Element e = it.next();
      e.set(f.apply(e.get(), y));
    }
    invalidateCachedLength();
    return this;
  }

  @Override
  public Vector assign(DoubleFunction function) {
    Iterator<Element> it = function.apply(0) == 0 ? iterateNonZero() : iterator();
    while (it.hasNext()) {
      Element e = it.next();
      e.set(function.apply(e.get()));
    }
    invalidateCachedLength();
    return this;
  }

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

    boolean isDensifying = Math.abs(function.apply(0.0, 0.0) - 0.0) > Constants.EPSILON;
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
    if (isDensifying) {
      thisIterator = this.iterator();
      thatIterator = other.iterator();
      while (thisIterator.hasNext() && thatIterator.hasNext()) {
        thisElement = thisIterator.next();
        thisElement.set(function.apply(thisElement.get(), thatIterator.next().get()));
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
    invalidateCachedLength();
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
