package org.apache.mahout.clustering.streaming.vectorizer;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class TFIDFVectorizer {
  // The scoring function that gets a (TF, DF) pair and computes the score.
  private Function<TFIDFScorer.Tuple, Double> tfIdfScorer;

  // Whether the resulting vectors should be normalized (so their length is 1). This should nearly always be true.
  private boolean normalize;

  public TFIDFVectorizer(Function<TFIDFScorer.Tuple, Double> tfIdfScorer, boolean normalize) {
    this.tfIdfScorer = tfIdfScorer;
    this.normalize = normalize;
  }

  /**
   * Vectorizes a list of paths writing the output to a given output path.
   * @param paths paths to documents to vectorize.
   * @param outputPath path to output sequence file containing the document vectors.
   * @param fs filesystem the paths are on.
   * @param conf Hadoop configuration.
   * @throws IOException
   */
  public void vectorizePaths(List<String> paths, Path outputPath, FileSystem fs, Configuration conf)
      throws IOException {
    SequenceFile.Writer writer =
        SequenceFile.createWriter(fs, conf, outputPath, Text.class, VectorWritable.class);
    Iterable<Vector> documentVectors = vectorize(paths);
    int i = 0;
    for (Vector documentVector : documentVectors) {
      String path = paths.get(i++);
      writer.append(new Text(path), new VectorWritable(documentVector));
    }
    writer.close();
  }

  /**
   * Vectorizes the documents from the given paths.
   * @param paths the paths to documents to vectorize.
   * @return an Iterable of the resulting vectors. The vectors are in the same order as the given paths.
   * @throws IOException
   */
  public Iterable<Vector> vectorize(final Iterable<String> paths) throws IOException {
    List<Map<String, Integer>> wordTFDictionaries = Lists.newArrayList();
    for (String path : paths) {
      wordTFDictionaries.add(buildWordTFDictionary(path));
    }
    return vectorize(wordTFDictionaries);
  }

  /**
   * Vectorizes the documents described by the list of frequency dictionaries.
   * @param wordTFDictionaries the list of word term frequency dictionaries to use.
   * @return an Iterable of the resulting vectors. The vectors are in the same order as the given dictionaries.
   */
  public Iterable<Vector> vectorize(final List<Map<String, Integer>> wordTFDictionaries) {
    // The word dictionary from Lucene tokens to the a pair containing the document frequency of the word and its
    // final position in the vectorized document.

    final Map<String, Pair<Integer, Integer>> wordDFDictionary = Maps.newHashMap();

    int numPaths = 0;
    // Build the document frequency dictionary for all the files.
    for (Map<String, Integer> wordTFDictionary : wordTFDictionaries) {
      for (Map.Entry<String, Integer> wordEntry : wordTFDictionary.entrySet()) {
        String word = wordEntry.getKey();
        Pair<Integer, Integer> dfValue = wordDFDictionary.get(word);
        if (dfValue == null) {
          wordDFDictionary.put(word, Pair.of(1, -1));
        } else {
          wordDFDictionary.put(word, Pair.of(dfValue.getFirst() + 1, -1));
        }
      }
      System.out.printf("Tokenized document %d\n", numPaths);
      ++numPaths;
    }
    int wordIndex = 0;
    for (Map.Entry<String, Pair<Integer, Integer>> entry : wordDFDictionary.entrySet()) {
      entry.setValue(Pair.of(entry.getValue().getFirst(), wordIndex++));
    }

    // Build the actual vectors.
    final int finalNumPaths = numPaths;
    return new Iterable<Vector>() {
      @Override
      public Iterator<Vector> iterator() {
        return new Iterator<Vector>() {
          private int i = 0;
          private int numWords = wordDFDictionary.size();

          @Override
          public boolean hasNext() {
            return i < finalNumPaths;
          }

          @Override
          public Vector next() {
            Map<String, Integer> wordTFDictionary = wordTFDictionaries.get(i++);
            Vector documentVector = new SequentialAccessSparseVector(numWords);
            for (Map.Entry<String, Integer> tfEntry : wordTFDictionary.entrySet()) {
              Pair<Integer, Integer> dfValue = wordDFDictionary.get(tfEntry.getKey());
              documentVector.set(dfValue.getSecond(),
                  tfIdfScorer.apply(new TFIDFScorer.Tuple(tfEntry.getValue(), dfValue.getFirst(),
                      finalNumPaths)));
            }
            System.out.printf("Vectorized document %d\n", i - 1);
            if (normalize) {
              return documentVector.normalize();
            }
            return documentVector;
          }

          @Override
          public void remove() {
            throw new UnsupportedOperationException();
          }
        };
      }
    };
  }

  /**
   * Builds a term frequency dictionary for the words in a file.
   * @param path the name of the file to be processed.
   * @return a map of words to frequency counts.
   */
  public static Map<String, Integer> buildWordTFDictionary(String path) throws IOException {
    return buildWordTFDictionary(new FileReader(path));
  }

  /**
   * Builds a term frequency dictionary for the words in a reader.
   * @param reader the reader to get the words from.
   * @return a map of words to frequency counts.
   */
  public static Map<String, Integer> buildWordTFDictionary(Reader reader) throws IOException {
    Tokenizer tokenizer = new StandardTokenizer(Version.LUCENE_41, reader);
    tokenizer.reset();
    CharTermAttribute cattr = tokenizer.addAttribute(CharTermAttribute.class);
    Map<String, Integer> wordTFDictionary = Maps.newHashMap();
    while (tokenizer.incrementToken()) {
      String word = cattr.toString();
      Integer tf = wordTFDictionary.get(word);
      if (tf == null) {
        wordTFDictionary.put(word, 1);
      } else {
        wordTFDictionary.put(word, tf + 1);
      }
    }
    tokenizer.end();
    tokenizer.close();
    return wordTFDictionary;
  }
}

