package org.apache.mahout.math.random;

import java.util.Random;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.jet.random.Normal;
import org.apache.mahout.math.jet.random.Uniform;

public class RandomProjector {
  final static Random random = RandomUtils.getRandom(System.currentTimeMillis());

  /**
   * Generates a basis matrix of size projectedVectorSize x vectorSize. Multiplying a a vector by
   * this matrix results in the projected vector.
   *
   * The rows of the matrix are sampled from a multi normal distribution.
   *
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a projection matrix
   */
  public static Matrix generateBasisNormal(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    basisMatrix.assign(new Normal(0.0, 1.0, random));
    for (MatrixSlice row : basisMatrix) {
      row.vector().assign(row.normalize());
    }
    return basisMatrix;
  }

  public static Matrix generateBasisUniform(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    basisMatrix.assign(new Uniform(0.0, 1.0, random));
    for (MatrixSlice row : basisMatrix) {
      row.vector().assign(row.normalize());
    }
    return basisMatrix;
  }

  /**
   * Generates a basis matrix of size projectedVectorSize x vectorSize. Multiplying a a vector by
   * this matrix results in the projected vector.
   *
   * The rows of a matrix are sample from a distribution where:
   * - +1 has probability 1/2,
   * - -1 has probability 1/2
   *
   * See Achlioptas, D. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins.
   * Journal of Computer and System Sciences, 66(4), 671–687. doi:10.1016/S0022-0000(03)00025-4
   *
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a projection matrix
   */
  public static Matrix generateBasisPlusMinusOne(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    for (int i = 0; i < projectedVectorSize; ++i) {
      for (int j = 0; j < vectorSize; ++j) {
        basisMatrix.set(i, j, random.nextInt(2) == 0 ? +1 : -1);
      }
    }
    for (MatrixSlice row : basisMatrix) {
      row.vector().assign(row.normalize());
    }
    return basisMatrix;
  }

  /**
   * Generates a basis matrix of size projectedVectorSize x vectorSize. Multiplying a a vector by
   * this matrix results in the projected vector.
   *
   * The rows of a matrix are sample from a distribution where:
   * - 0 has probability 2/3,
   * - +1 has probability 1/6,
   * - -1 has probability 1/6
   *
   * See Achlioptas, D. (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins.
   * Journal of Computer and System Sciences, 66(4), 671–687. doi:10.1016/S0022-0000(03)00025-4
   *
   * @param projectedVectorSize final projected size of a vector (number of projection vectors)
   * @param vectorSize initial vector size
   * @return a projection matrix
   */
  public static Matrix generateBasisZeroPlusMinusOne(int projectedVectorSize, int vectorSize) {
    Matrix basisMatrix = new DenseMatrix(projectedVectorSize, vectorSize);
    Multinomial<Double> choice = new Multinomial<Double>();
    choice.add(0.0, 2/3.0);
    choice.add(+Math.sqrt(3.0), 1 / 6.0);
    choice.add(-Math.sqrt(3.0), 1 / 6.0);
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
}
