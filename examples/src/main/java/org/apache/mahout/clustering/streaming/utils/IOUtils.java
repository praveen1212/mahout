package org.apache.mahout.clustering.streaming.utils;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.streaming.mapreduce.CentroidWritable;
import org.apache.mahout.math.neighborhood.ProjectionSearch;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

public class IOUtils {
  /**
   * Writes out the list of datapoints (or centroids) as a CSV file. The resulting file has a header with the same number
   * of columns as the dimensionality of the datapoints, d, and with names x0 to x{d-1}.
   * @param datapoints the vectors to dump.
   * @param outPath the destination path.
   * @throws java.io.FileNotFoundException
   */
  public static void generateCSVFromVectors(Iterable<? extends Vector> datapoints,
                                            Configuration conf, String outPath) throws IOException {
    int numDimensions = -1;
    int numDatapoints = 0;
    System.out.printf("Creating %s\n", outPath);
    PrintStream outputStream = new PrintStream(FileSystem.get(conf).create(new Path(outPath)));
    for (Vector datapoint : datapoints) {
      if (numDimensions < 0) {
        numDimensions = datapoint.size();
        for (int i = 0; i < numDimensions; ++i) {
          outputStream.printf("x%d", i);
          if (i < numDimensions - 1) {
            outputStream.printf(", ");
          } else {
            outputStream.println();
          }
        }
      }
      int numNondefault = datapoint.getNumNondefaultElements();
      for (Vector.Element vi : datapoint) {
        outputStream.printf("%f", vi.get());
        if (numNondefault > 1) {
          outputStream.printf(", ");
          --numNondefault;
        } else {
          outputStream.println();
        }
      }
      ++numDatapoints;
    }
    System.out.printf("Done. Wrote %d datapoints\n", numDatapoints);
    outputStream.close();
  }

  /**
   * Generates a MMF (Matrix Market File: http://math.nist.gov/MatrixMarket/formats.html).
   * The file is written in coordinate format because the vectors are expected to be sparse.
   * @param datapoints the datapoints that form the matrix.
   * @param outPath the path to the resulting MMF file.
   * @throws FileNotFoundException
   */
  public static void generateMMFromVectors(Iterable<? extends Vector> datapoints,
                                           String outPath) throws FileNotFoundException {
    PrintStream  outputStream = new PrintStream(new FileOutputStream(outPath));
    outputStream.printf("%%%%MatrixMarket matrix coordinate real general\n");
    int numNonZero = 0;
    int numDimensions = -1;
    int numDatapoints = 0;
    for (Vector datapoint : datapoints) {
      if (numDimensions < 0) {
        numDimensions = datapoint.size();
      }
      numNonZero += datapoint.getNumNondefaultElements();
      ++numDatapoints;
    }
    outputStream.printf("%d %d %d\n", numDatapoints, numDimensions, numNonZero);
    int i = 0;
    for (Vector datapoint : datapoints) {
      for (int j = 0; j < numDimensions; ++j) {
        double coord = datapoint.get(j);
        if (coord != 0) {
          outputStream.printf("%d %d %f\n", i + 1, j + 1, coord);
        }
      }
      ++i;
    }
    outputStream.close();
  }

  /**
   * Writes centroids to a sequence file.
   * @param centroids the centroids to write.
   * @param conf the configuration for the HDFS to write the file to.
   * @param name the path of the output file.
   * @throws IOException
   */
  public static void writeCentroidsToSequenceFile(Iterable<Centroid> centroids, Configuration conf, String name) throws IOException {
    SequenceFile.Writer writer = SequenceFile.createWriter(FileSystem.get(conf), conf,
        new Path(name), IntWritable.class, CentroidWritable.class);
    int i = 0;
    for (Centroid centroid : centroids) {
      writer.append(new IntWritable(i++), new CentroidWritable(centroid));
    }
    writer.close();
  }

  /**
   * Converts the VectorWritable values in a sequence file into Centroids lazily.
   * @param dirIterable the source iterable (comes from a SequenceFileDirIterable).
   * @return an Iterable<Centroid> with the converted vectors.
   */
  public static Iterable<Centroid> getCentroidsFromPairIterable(
      Iterable<Pair<Text, VectorWritable>> dirIterable) {
    return Iterables.transform(dirIterable, new Function<Pair<Text, VectorWritable>, Centroid>() {
      private int count = 0;

      @Override
      public Centroid apply(Pair<Text, VectorWritable> input) {
        Preconditions.checkNotNull(input);
        return new Centroid(count++, input.getSecond().get().clone(), 1);
      }
    });
  }

  /**
   * Converts CentroidWritable values in a sequence file into Centroids lazily.
   * @param dirIterable the source iterable (comes from a SequenceFileDirIterable).
   * @return an Iterable<Centroid> with the converted vectors.
   */
  public static Iterable<Centroid> getCentroidsFromCentroidWritableIterable(
      Iterable<CentroidWritable>  dirIterable) {
    return Iterables.transform(dirIterable, new Function<CentroidWritable, Centroid>() {
      @Override
      public Centroid apply(CentroidWritable input) {
        Preconditions.checkNotNull(input);
        return input.getCentroid().clone();
      }
    });
  }

  /**
   * Converts VectorWritable values in a sequence file into Vectors lazily.
   * @param dirIterable the source iterable (comes from a SequenceFileDirIterable).
   * @return an Iterable<Vector> with the converted vectors.
   */
  public static Iterable<Vector> getVectorsFromVectorWritableIterable(Iterable<VectorWritable> dirIterable) {
    return Iterables.transform(dirIterable, new Function<VectorWritable, Vector>() {
      @Override
      public Vector apply(VectorWritable input) {
        Preconditions.checkNotNull(input);
        return input.get().clone();
      }
    });
  }

  /**
   * Reads the vectors from inPath and randomly projects them down to reducedDimension dimensions.
   * The inputPaths, inputVectors and reducedVectors arguments are output arguments and will be populated by the method.
   * @param inPath the seqfile with vectors to read
   * @param reducedDimension the dimension down to which the vectors must be projected
   * @param inputPaths a list of strings that will be populated (and that should initially be empty) containing the
   *                   key (path) for each vector (document)
   * @param inputVectors a list of the vectors as read from the seqfile
   * @param reducedVectors the list of projected vectors
   * @throws java.io.IOException
   */
  @Deprecated
  public static void getInputVectors(String inPath, int reducedDimension,
                                     List<String> inputPaths,
                                     List<Vector> inputVectors,
                                     List<Centroid> reducedVectors) throws IOException {
    System.out.println("Started reading data");
    Path inFile = new Path(inPath);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, inFile, conf);
    Text key = new Text();
    VectorWritable value = new VectorWritable();

    double start = System.currentTimeMillis();
    while (reader.next(key, value)) {
      inputPaths.add(key.toString());
      inputVectors.add(value.get().clone());
    }

    int initialDimension = inputVectors.get(0).size();
    Matrix basisMatrix = ProjectionSearch.generateBasis(reducedDimension, initialDimension);
    int numVectors = 0;
    for (Vector v : inputVectors) {
      reducedVectors.add(new Centroid(numVectors++, basisMatrix.times(v), 1));
    }
    double end = System.currentTimeMillis();
    System.out.printf("Finished reading data; initial dimension %d; projected to %d; took %f\n",
        initialDimension, reducedDimension, end - start);
  }

  public static Pair<List<String>, List<Centroid>> getKeysAndVectors(String inPath, int projectionDimension) throws IOException {
    return getKeysAndVectors(inPath, projectionDimension, Integer.MAX_VALUE);
  }

  /**
   * Reads in a seqfile mapping keys (file paths as strings) to vectors (documents) and projects the vectors to a given
   * dimension.
   * @param inPath the input path to the seqfile.
   * @param projectionDimension the dimension to project to.
   * @return a pair of List<String> and List<Centroid>, the first list containing the values of the keys and the second
   * list containing the projected vectors as centroids.
   * @throws java.io.IOException
   */
  public static Pair<List<String>, List<Centroid>> getKeysAndVectors(String inPath, int projectionDimension, int limit) throws IOException {
    System.out.printf("Started reading data\n");

    Path inFile = new Path(inPath);
    Configuration conf = new Configuration();

    Matrix projectionMatrix = null;
    Pair<List<String>, List<Centroid>> result =
        new Pair<List<String>, List<Centroid>>(new ArrayList<String>(), new ArrayList<Centroid>());
    int numVectors = 0;

    double start = System.currentTimeMillis();
    SequenceFileDirIterable<Text, VectorWritable> dirIterable =
        new SequenceFileDirIterable<Text, VectorWritable>(inFile, PathType.LIST, conf);
    for (Pair<Text, VectorWritable> entry : dirIterable) {
      if (projectionMatrix == null) {
        projectionMatrix = ProjectionSearch.generateBasis(projectionDimension, entry.getSecond().get().size());
      }
      result.getFirst().add(entry.getFirst().toString());
      result.getSecond().add(new Centroid(numVectors++, projectionMatrix.times(entry.getSecond().get()), 1));
      --limit;
      if (limit == 0) {
        break;
      }
    }
    double end = System.currentTimeMillis();

    System.out.printf("Finished reading data: took %f [s]\n", (end - start) / 1000);
    return result;
  }
}
