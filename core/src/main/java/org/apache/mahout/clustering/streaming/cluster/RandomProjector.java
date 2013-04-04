package org.apache.mahout.clustering.streaming.cluster;

import com.google.common.collect.Lists;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.random.Multinomial;
import org.apache.mahout.math.random.Normal;

import java.util.List;

public class RandomProjector {
  /**
   * Generates a basis matrix of size projectedVectorSize x vectorSize. Multiplying a a vector by
   * this matrix results in the projected vector.
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a projection matrix
   */
  public static Matrix generateBasisNormal(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    basisMatrix.assign(new Normal());
    for (MatrixSlice row : basisMatrix) {
      row.vector().assign(row.normalize());
    }
    return basisMatrix;
  }

  public static Matrix generateBasisZeroPlusMinusOne(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    Multinomial<Integer> choice = new Multinomial<Integer>();
    choice.add(0, 2/3.0);
    choice.add(+1, 1/6.0);
    choice.add(-1, 1/6.0);
    for (int i = 0; i < projectedVectorSize; ++i) {
      for (int j = 0; j < vectorSize; ++j) {
        basisMatrix.set(i, j, choice.sample());
      }
    }
    for (MatrixSlice row : basisMatrix) {
      row.vector().assign(row.normalize());
    }
    return basisMatrix;
  }

  public static Matrix generateBasisPlusMinusOne(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    for (int i = 0; i < projectedVectorSize; ++i) {
      for (int j = 0; j < vectorSize; ++j) {
        basisMatrix.set(i, j, RandomUtils.nextInt(2) == 0 ? +1 : -1);
      }
    }
    for (MatrixSlice row : basisMatrix) {
      row.vector().assign(row.normalize());
    }
    return basisMatrix;
  }

  /**
   * Generates a list of projectedVectorSize vectors, each of size vectorSize. This looks like a
   * matrix of size (projectedVectorSize, vectorSize).
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a list of projection vectors
   */
  public static List<Vector> generateVectorBasis(int projectedVectorSize, int vectorSize) {
    final DoubleFunction random = new Normal();
    List<Vector> basisVectors = Lists.newArrayList();
    for (int i = 0; i < projectedVectorSize; ++i) {
      Vector basisVector = new DenseVector(vectorSize);
      basisVector.assign(random);
      basisVector.normalize();
      basisVectors.add(basisVector);
    }
    return basisVectors;
  }
}
