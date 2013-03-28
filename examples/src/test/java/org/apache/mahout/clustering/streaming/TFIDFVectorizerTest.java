package org.apache.mahout.clustering.streaming;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.clustering.streaming.vectorizer.TFIDFScorer;
import org.apache.mahout.clustering.streaming.vectorizer.TFIDFVectorizer;
import org.apache.mahout.math.Vector;
import org.hamcrest.CoreMatchers;
import org.hamcrest.MatcherAssert;
import org.junit.Test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public class TFIDFVectorizerTest {
  @Test
  public void testSimpleFile() throws IOException {
    TFIDFVectorizer vectorizer = new TFIDFVectorizer(new TFIDFScorer.Linear(), true);
    Map<String, Integer> wordCounts = Maps.newHashMap();
    wordCounts.put("alfa", 23);
    wordCounts.put("beta", 2);
    wordCounts.put("gamma", 40);
    wordCounts.put("delta", 1);
    File tempFile = File.createTempFile("tfidf-vectorizer-test", ".txt");
    FileWriter writer = new FileWriter(tempFile);
    for (Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
      for (int i = 0;  i < entry.getValue(); ++i) {
        writer.write(entry.getKey() + " ");
      }
      writer.write('\n');
    }
    writer.close();

    Map<String, Integer> actualCounts = vectorizer.buildWordTFDictionary(tempFile.getPath());
    MatcherAssert.assertThat(actualCounts, CoreMatchers.is(wordCounts));

    List<String> pathList = Lists.newArrayList();
    pathList.add(tempFile.getPath());

    Iterable<Vector> vectors = vectorizer.vectorize(pathList);
    for (Vector vector : vectors) {
      System.out.println(vector.toString());
    }

    FileUtils.forceDeleteOnExit(tempFile);
  }
}
