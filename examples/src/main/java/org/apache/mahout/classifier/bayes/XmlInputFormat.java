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

package org.apache.mahout.classifier.bayes;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;

import java.io.IOException;

/**
 * Reads records that are delimited by a specifc begin/end tag.
 */
public class XmlInputFormat extends TextInputFormat {
    public static final String START_TAG_KEY = "xmlinput.start";
    public static final String END_TAG_KEY = "xmlinput.end";

    @SuppressWarnings("unchecked")
    public RecordReader getRecordReader(InputSplit inputSplit, JobConf jobConf, Reporter reporter) throws IOException {
        return new XmlRecordReader((FileSplit) inputSplit, jobConf);
    }

    public static class XmlRecordReader implements RecordReader<WritableComparable,Writable> {
        private byte[] startTag;
        private byte[] endTag;
        private long start;
        private long end;
        private FSDataInputStream fsin;
        private final DataOutputBuffer buffer = new DataOutputBuffer();

        public XmlRecordReader(FileSplit split, JobConf jobConf) throws IOException {
            startTag = jobConf.get("xmlinput.start").getBytes("utf-8");
            endTag = jobConf.get("xmlinput.end").getBytes("utf-8");

            // open the file and seek to the start of the split
            start = split.getStart();
            end = start + split.getLength();
            Path file = split.getPath();
            FileSystem fs = file.getFileSystem(jobConf);
            fsin = fs.open(split.getPath());
            fsin.seek(start);
        }

        public boolean next(WritableComparable key, Writable value) throws IOException {
            if (fsin.getPos() < end) {
                if (readUntilMatch(startTag, false)) {
                    try {
                        buffer.write(startTag);
                        if (readUntilMatch(endTag, true)) {
                            ((Text) key).set(Long.toString(fsin.getPos()));
                            ((Text) value).set(buffer.getData(), 0, buffer.getLength());
                            return true;
                        }
                    }
                    finally {
                        buffer.reset();
                    }
                }
            }
            return false;
        }

        public WritableComparable createKey() {
            return new Text();
        }

        public Writable createValue() {
            return new Text();
        }

        public long getPos() throws IOException {
            return fsin.getPos();
        }

        public void close() throws IOException {
            fsin.close();
        }

        public float getProgress() throws IOException {
            return ((float) (fsin.getPos() - start)) / ((float) (end - start));
        }

        /////////////////////////////////////////////////

        private boolean readUntilMatch(byte[] match, boolean withinBlock) throws IOException {
            int i = 0;
            while (true) {
                int b = fsin.read();
                // end of file:
                if (b == -1) return false;
                // save to buffer:
                if (withinBlock) buffer.write(b);

                // check if we're matching:
                if (b == match[i]) {
                    i++;
                    if (i >= match.length) return true;
                } else i = 0;
                // see if we've passed the stop point:
                if(!withinBlock && i == 0 && fsin.getPos() >= end) return false;
            }
        }
    }
} 