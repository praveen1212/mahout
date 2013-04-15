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

package org.apache.mahout.math;

import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.Iterator;
import java.util.Random;

public final class TestSequentialAccessSparseVector extends AbstractVectorTest<SequentialAccessSparseVector> {

  @Override
  Vector generateTestVector(int cardinality) {
    return new SequentialAccessSparseVector(cardinality);
  }

  @Test
  public void testDotSuperBig() {
    Vector w = new SequentialAccessSparseVector(Integer.MAX_VALUE, 12);
    w.set(1, 0.4);
    w.set(2, 0.4);
    w.set(3, -0.666666667);

    Vector v = new SequentialAccessSparseVector(Integer.MAX_VALUE, 12);
    v.set(3, 1);

    assertEquals("super-big", -0.666666667, v.dot(w), EPSILON);
  }


  @Override
  public SequentialAccessSparseVector vectorToTest(int size) {
    SequentialAccessSparseVector r = new SequentialAccessSparseVector(size);
    Random gen = RandomUtils.getRandom();
    for (int i = 0; i < 3; i++) {
      r.set(gen.nextInt(r.size()), gen.nextGaussian());
    }
    return r;
  }

  @Test
  public void testDenseVectorIteration() {
    vectorIterationTest(new DenseVector(100));
  }

  @Test
  public void testSequentialAccessSparseVector() {
    vectorIterationTest(new SequentialAccessSparseVector(100));
  }

  @Test
  public void testRandomAccessSparseVector() {
    vectorIterationTest(new RandomAccessSparseVector(100));
  }

  public void vectorIterationTest(Vector vector) {
    System.out.printf("%s\n", vector.getClass().toString());
    vector.set(0, 1);
    vector.set(2, 2);
    vector.set(4, 3);
    vector.set(6, 4);
    Iterator<Vector.Element> vectorIterator = vector.iterateNonZero();
    Vector.Element element = null;
    int i = 0;
    while (vectorIterator.hasNext()) {
      if (i % 2 == 0) {
        element = vectorIterator.next();
        System.out.printf("Advancing\n");
      }
      System.out.printf("%d %d %f\n", i, element.index(), element.get());
      ++i;
    }
    System.out.printf("Done\n\n");
  }
}