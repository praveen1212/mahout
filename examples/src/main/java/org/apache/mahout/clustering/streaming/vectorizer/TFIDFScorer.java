package org.apache.mahout.clustering.streaming.vectorizer;

import com.google.common.base.Function;

public class TFIDFScorer {
  static public class Tuple {
    double tf;
    double df;
    double n;

    public Tuple(int tf, int df, int n) {
      this.tf = tf;
      this.df = df;
      this.n = n;
    }
  }

  static public class Linear implements Function<Tuple, Double> {
    @Override
    public Double apply(Tuple input) {
      return input.tf * (input.n > 1 ? Math.log(input.n / input.df) : 1);
    }
  }

  static public class Const implements Function<Tuple, Double> {
    @Override
    public Double apply(Tuple input) {
      return (input.n > 1 ? Math.log(input.n / input.df) : 1);
    }
  }

  static public class Log implements Function<Tuple, Double> {
    @Override
    public Double apply(Tuple input) {
      return Math.log(input.tf) * (input.n > 1 ? Math.log(input.n / input.df) : 1);
    }
  }

  static public class Sqrt implements Function<Tuple, Double> {
    @Override
    public Double apply(Tuple input) {
      return Math.sqrt(input.tf) * (input.n > 1 ? Math.log(input.n / input.df) : 1);
    }
  }
}
