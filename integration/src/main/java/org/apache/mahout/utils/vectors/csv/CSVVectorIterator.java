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

package org.apache.mahout.utils.vectors.csv;

import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Lists;
import org.apache.commons.csv.CSVParser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.List;

/**
 * Iterates a CSV file and produces {@link org.apache.mahout.math.Vector}.
 * <br/>
 * The Iterator returned throws {@link UnsupportedOperationException} for the {@link java.util.Iterator#remove()}
 * method.
 * <p/>
 * Assumes DenseVector for now, but in the future may have the option of mapping columns to sparse format
 * <p/>
 *
 * Non-numeric attributes are IGNORED. If categorical attributes are required, code them appropriately first.
 * The Iterator is not thread-safe.
 */
public class CSVVectorIterator extends AbstractIterator<Vector> {

  private final CSVParser parser;

  private boolean hasHeaderLine;

  private List<String> header;

  public CSVVectorIterator(boolean hasHeaderLine, Path path, Configuration conf) throws IOException {
    this.parser = new CSVParser(new InputStreamReader(new DataInputStream(FileSystem.get(conf).open(path))));
    this.hasHeaderLine = hasHeaderLine;
    if (hasHeaderLine) {
      initHeader();
    }
  }

  public CSVVectorIterator(boolean hasHeaderLine, Reader reader) {
    this.parser = new CSVParser(reader);
    this.hasHeaderLine = hasHeaderLine;
    if (hasHeaderLine) {
      initHeader();
    }
  }

  @Override
  protected Vector computeNext() {
    String[] line;
    try {
      line = parser.getLine();
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    if (line == null) {
      return endOfData();
    }
    Vector result = new DenseVector(line.length);
    for (int i = 0; i < line.length; i++) {
      try {
        result.setQuick(i, Double.parseDouble(line[i]));
      } catch (NumberFormatException e) {
        // This field is simply ignored if the value is not numeric.
        // Ideally, a more flexible processor should support 1 to K encoding for categorical features,
        // but currently this step needs to be done outside Mahout.
      }
    }
    return result;
  }

  private void initHeader() {
    if (header == null && hasHeaderLine) {
      try {
        String[] line = parser.getLine();
        if (line != null) {
          header = Lists.newArrayList(line);
        }
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    }
  }

  public List<String> getHeader() {
    return header;
  }
}
