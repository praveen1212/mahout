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

import org.apache.mahout.math.function.DoubleDoubleFunction;

/**
 * A centroid is a weighted vector.  We have it delegate to the vector itself for lots of operations
 * to make it easy to use vector search classes and such.
 */
public class Centroid extends WeightedVector {
  public Centroid(WeightedVector original) {
    super(original.getVector().like().assign(original), original.getWeight(), original.getIndex());
  }

  public Centroid(int key, Vector initialValue) {
    super(initialValue, 1, key);
  }

  public Centroid(int key, Vector initialValue, double weight) {
    super(initialValue, weight, key);
  }

  public static Centroid create(int key, Vector initialValue) {
    if (initialValue instanceof WeightedVector) {
      return new Centroid(key, new DenseVector(initialValue), ((WeightedVector) initialValue).getWeight());
    } else {
      return new Centroid(key, new DenseVector(initialValue), 1);
    }
  }

  public void update(Vector v) {
    if (v instanceof Centroid) {
      Centroid c = (Centroid) v;
      update(c.delegate, c.getWeight());
    } else {
      update(v, 1);
    }
  }

  public void update(Vector other, final double wy) {
    final double wx = getWeight();
    final double tw = wx + wy;
    delegate.assign(other, new DoubleDoubleFunction() {
      @Override
      public double apply(double x, double y) {
        return (wx * x + wy * y) / tw;
      }

      /**
       * f(x, 0) = wx * x / tw = x iff wx = tw (practically, impossible, as tw = wx + wy and wy > 0)
       * @return true iff f(x, 0) = x for any x
       */
      @Override
      public boolean isLikeRightPlus() {
        return wx == tw;
      }

      /**
       * f(0, y) = wy * y / tw = 0 iff y = 0
       * @return true iff f(0, y) = 0 for any y
       */
      @Override
      public boolean isLikeLeftMult() {
        return false;
      }

      /**
       * f(x, 0) = wx * x / tw = 0 iff x = 0
       * @return true iff f(x, 0) = 0 for any x
       */
      @Override
      public boolean isLikeRightMult() {
        return false;
      }

      /**
       * wx * x + wy * y = wx * y + wy * x iff wx = wy
       * @return true iff f(x, y) = f(y, x) for any x, y
       */
      @Override
      public boolean isCommutative() {
        return wx == wy;
      }

      /**
       * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
       */
      @Override
      public boolean isAssociative() {
        return false;
      }
    });
    setWeight(tw);
  }

  @Override
  public Centroid like() {
    return new Centroid(getIndex(), getVector().like(), getWeight());
  }

  /**
   * Gets the index of this centroid.  Use getIndex instead to maintain standard names.
   */
  @Deprecated
  public int getKey() {
    return getIndex();
  }

  public void addWeight(double newWeight) {
    setWeight(getWeight() + newWeight);
  }

  @Override
  public String toString() {
    return String.format("key = %d, weight = %.2f, vector = %s", getIndex(), getWeight(), delegate);
  }

  @Override
  public Centroid clone() {
    return new Centroid(this);
  }
}
