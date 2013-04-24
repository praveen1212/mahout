package org.apache.mahout.math;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;

import java.util.Random;

public class LargeVectorsTest {
  private static final int CARDINALITY = 1 << 14;
  private static final int NUM_NONDEFAULT = 1 << 10;
  private static final Random random = RandomUtils.getRandom();

  public SequentialAccessSparseVector getSasv() {
    return (SequentialAccessSparseVector) initialize(new SequentialAccessSparseVector(CARDINALITY, NUM_NONDEFAULT),
        NUM_NONDEFAULT);
  }

  public RandomAccessSparseVector getRasv() {
    return (RandomAccessSparseVector) initialize(new RandomAccessSparseVector(CARDINALITY, NUM_NONDEFAULT),
        NUM_NONDEFAULT);
  }

  public DenseVector getDv(int size) {
    return (DenseVector) initialize(new DenseVector(size), size);
  }

  public Vector initialize(Vector vector, int numNondefault) {
    int cardinality = vector.size();
    System.out.printf("cardinality %d numNondefault %d\n", cardinality, numNondefault);
    for (int i = 0; i < numNondefault; ++i) {
      vector.setQuick(random.nextInt(cardinality), random.nextGaussian());
    }
    return vector;
  }

  @Test
  public void testSparseDense() {
    testOperations(getSasv(), getDv(CARDINALITY));
  }

  @Test
  public void testSparseSparse() {
    testOperations(getSasv(), getSasv());
  }

  @Test
  public void testSparseRandom() {
    testOperations(getSasv(), getRasv());
  }

  @Test
  public void testRandomDense() {
    testOperations(getRasv(), getDv(CARDINALITY));
  }

  @Test
  public void testRandomRandom() {
    testOperations(getRasv(), getRasv());
  }

  @Test
  public void testDenseDense() {
    testOperations(getDv(NUM_NONDEFAULT), getDv(NUM_NONDEFAULT));
  }

  public void testOperations(Vector x, Vector y) {
    long start = System.currentTimeMillis();
    testOperationsInternal(x, y);
    testOperationsInternal(y, x);
    long end = System.currentTimeMillis();
    System.out.printf(">> TIME %f [s]<<\n", (end - start) / 1000.0);
  }

  public void testOperationsInternal(Vector x, Vector y) {
    System.out.printf("x: class %s isSequential %s numNondefault %d isAddConstant %s iterateNonzero %f randomAcess %f\n"
    + "y: class %s isSequential %s numNondefault %d isAddConstant %s iterateNonzero %f randomAcess %f\n",
        x.getClass().toString(), x.isSequentialAccess(), x.getNumNondefaultElements(),
        x.isAddConstantTime(), x.getIterateNonzeroAdvanceTime(), x.getRandomAccessLookupTime(),
        y.getClass().toString(), y.isSequentialAccess(), y.getNumNondefaultElements(),
        y.isAddConstantTime(), y.getIterateNonzeroAdvanceTime(), y.getRandomAccessLookupTime());
    System.out.printf("PLUS\n");
    x.plus(y);
    System.out.printf("MINUS\n");
    x.minus(y);
    System.out.printf("TIMES\n");
    x.times(y);
    System.out.printf("DOT\n");
    x.dot(y);
    System.out.printf("DISTANCE SQUARED\n");
    x.getDistanceSquared(y);
    System.out.printf("MINUS_SQUARED\n");
    Vector xc = x.like();
    xc.assign(x);
    xc.assign(y, Functions.MINUS_SQUARED);
    System.out.printf("SECOND\n");
    xc.assign(x);
    xc.assign(y, Functions.SECOND);
    System.out.printf("PLUS_ABS\n");
    xc.assign(x);
    xc.assign(y, Functions.PLUS_ABS);
    System.out.println();
  }
}
