package org.apache.mahout.utils.vectors.csv;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.Iterator;

public class CSVVectorIterable implements Iterable<Vector> {
  private Reader reader;

  public CSVVectorIterable(Path path, Configuration conf) throws IOException {
    this.reader = new InputStreamReader(new DataInputStream(FileSystem.get(conf).open(path)));
  }

  public CSVVectorIterable(Reader reader) {
    this.reader = reader;
  }

  @Override
  public Iterator<Vector> iterator() {
    return new CSVVectorIterator(reader);
  }
}
