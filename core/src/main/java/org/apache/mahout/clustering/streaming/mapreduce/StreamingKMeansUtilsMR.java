package org.apache.mahout.clustering.streaming.mapreduce;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.neighborhood.*;
import org.slf4j.Logger;

public class StreamingKMeansUtilsMR {

  public static UpdatableSearcher searcherFromConfiguration(Configuration conf, Logger log) {
    DistanceMeasure distanceMeasure;
    String distanceMeasureClass = conf.get(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    try {
      distanceMeasure = (DistanceMeasure)Class.forName(distanceMeasureClass).newInstance();
    } catch (Exception e) {
      log.error("Failed to instantiate distanceMeasure", e);
      throw new RuntimeException("Failed to instantiate distanceMeasure", e);
    }

    int numProjections =  conf.getInt(StreamingKMeansDriver.NUM_PROJECTIONS_OPTION, 20);
    int searchSize =  conf.getInt(StreamingKMeansDriver.SEARCH_SIZE_OPTION, 10);

    String searcherClass = conf.get(StreamingKMeansDriver.SEARCHER_CLASS_OPTION);

    if (searcherClass.equals(BruteSearch.class.getName())) {
      return ClassUtils.instantiateAs(searcherClass, UpdatableSearcher.class,
          new Class[]{DistanceMeasure.class}, new Object[]{distanceMeasure});
    } else if (searcherClass.equals(FastProjectionSearch.class.getName()) ||
        searcherClass.equals(ProjectionSearch.class.getName())) {
      return ClassUtils.instantiateAs(searcherClass, UpdatableSearcher.class,
          new Class[]{DistanceMeasure.class, int.class, int.class},
          new Object[]{distanceMeasure, numProjections, searchSize});
    } else if (searcherClass.equals(LocalitySensitiveHashSearch.class.getName())) {
      return ClassUtils.instantiateAs(searcherClass, UpdatableSearcher.class,
          new Class[]{DistanceMeasure.class, int.class},
          new Object[]{distanceMeasure, searchSize});
    } else {
      log.error("Unknown searcher class instantiation requested {}", searcherClass);
      throw new IllegalStateException("Unknown class instantiation requested");
    }
  }
}
