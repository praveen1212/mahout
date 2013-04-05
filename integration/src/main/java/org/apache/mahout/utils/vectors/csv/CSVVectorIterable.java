package org.apache.mahout.utils.vectors.csv;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

public class CSVVectorIterable implements Iterable<Vector> {
  private boolean hasHeaderLine;

  private Path path;

  private Configuration conf;

  private List<String> header;

  public CSVVectorIterable(boolean hasHeaderLine, Path path, Configuration conf) throws IOException {
    this.hasHeaderLine = hasHeaderLine;
    this.path = path;
    this.conf = conf;
    if (hasHeaderLine) {
      header = new CSVVectorIterator(hasHeaderLine, path, conf).getHeader();
    }
  }

  @Override
  public Iterator<Vector> iterator() {
    try {
      return new CSVVectorIterator(hasHeaderLine, path, conf);
    } catch (IOException e) {
      throw new IllegalStateException(e.getMessage(), e);
    }
  }

  public List<String> getHeader() {
    return header;
  }
}
